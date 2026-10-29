import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, ConcatDataset
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image
import shutil
import json

from accelerate.utils import set_seed
from utils.clip_model import ClipModel
import matplotlib.pyplot as plt
from utils.MINIM import MINIMGenerator, MINIMTrainer

# HB-FCL: High-fidelity Boundary-aware Federated Continual Learning Framework for Medical Image Diagnosis

tau = 2


def get_norm_and_transform():
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=3
            ),  # Keep a consistent channel count.
            transforms.Resize((224, 224)),  # Resize to a fixed shape.
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Optional random crop.
            transforms.RandomRotation(10),  # Mild rotation without changing structure.
            # transforms.RandomHorizontalFlip(p=0.5),  # Optional horizontal flip.
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform


def get_eval_transform():
    """Deterministic preprocessing for confidence/pseudo-label filtering."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _collect_all_images(root, postfix=["png", "jpg", "jpeg", "JPEG"]):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    postfix = tuple(postfix)
    for dirpath, _, files in os.walk(root):
        files.sort()
        for f in files:
            if f.endswith(postfix):
                images.append(os.path.join(dirpath, f))
    return images


def _sample_indices_by_combo(labels, nums, seed=0):
    labels = np.asarray(labels)
    total = len(labels)
    if nums is None or nums >= total:
        return np.arange(total, dtype=np.int64)

    # Keep combo order by first appearance to support deterministic remainder assignment.
    combo_to_indices = {}
    combo_order = []
    for idx, label in enumerate(labels):
        combo = tuple(int(v) for v in np.asarray(label).tolist())
        if combo not in combo_to_indices:
            combo_to_indices[combo] = []
            combo_order.append(combo)
        combo_to_indices[combo].append(idx)

    rng = np.random.RandomState(int(seed))
    for combo in combo_order:
        combo_indices = np.asarray(combo_to_indices[combo], dtype=np.int64)
        combo_to_indices[combo] = combo_indices[
            rng.permutation(len(combo_indices))
        ].tolist()

    n_combo = len(combo_order)
    per_combo = nums // n_combo
    remainder = nums % n_combo

    selected = []
    for i, combo in enumerate(combo_order):
        quota = per_combo + (1 if i < remainder else 0)
        take = min(quota, len(combo_to_indices[combo]))
        selected.extend(combo_to_indices[combo][:take])
        combo_to_indices[combo] = combo_to_indices[combo][take:]

    # If any combo is short, top-up from remaining pools in combo order.
    if len(selected) < nums:
        need = nums - len(selected)
        for combo in combo_order:
            if need <= 0:
                break
            if len(combo_to_indices[combo]) == 0:
                continue
            take = min(need, len(combo_to_indices[combo]))
            selected.extend(combo_to_indices[combo][:take])
            need -= take

    return np.asarray(selected, dtype=np.int64)


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data


class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, nums=None, sample_seed=0):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(
            self.root
        )  # [ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

        # Try to load the label file.
        labels_file = os.path.join(self.root, "labels.npy")
        if os.path.exists(labels_file):
            self.labels = np.load(labels_file)
            # Ensure image and label counts match.
            if len(self.labels) != len(self.images):
                raise ValueError(
                    f"Warning: Label count ({len(self.labels)}) != Image count ({len(self.images)}) in {self.root}"
                )
        else:
            raise ValueError(f"Labels file not found in {self.root}")

        # If needed, sample label combinations while preserving image-label alignment.
        if nums is not None and len(self.images) > nums:
            selected_idx = _sample_indices_by_combo(
                self.labels, int(nums), seed=sample_seed
            )
            self.images = [self.images[i] for i in selected_idx.tolist()]
            self.labels = self.labels[selected_idx]
            unique_combo_count = len(
                {tuple(int(v) for v in np.asarray(l).tolist()) for l in self.labels}
            )
            sample_info = (
                f"[LabeledImageDataset] random sampling: sample_seed={int(sample_seed)}, "
                f"selected={len(self.images)}, unique_combos={unique_combo_count}"
            )
            print(sample_info)

    def __getitem__(self, idx):
        with Image.open(self.images[idx]) as img:
            img = img.copy()
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            label = torch.from_numpy(self.labels[idx]).float()
            return img, label, len(label)
        else:
            return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        has_labels = "with labels" if self.labels is not None else "without labels"
        return (
            "Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s\n\t%s"
            % (self.root, len(self), self.transform, has_labels)
        )


class InMemoryLabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = list(images)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if not isinstance(label, torch.Tensor):
            label = torch.as_tensor(label, dtype=torch.float32)
        return image, label.float()


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification."""

    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()
        anti_targets = 1.0 - targets

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = anti_targets * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * anti_targets
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * anti_targets
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


class Ours(BaseLearner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self._network = IncrementalNet(args, False, False)
        self.r = args["r"]
        self.ltc = args["ltc"]
        self.tse_ltc = args["tse_ltc"]
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")

        self.clip_model = ClipModel(model_name="kaveh/rclip")
        self.label_embedding_cache = {}  # Maps label tuples to embeddings.
        self.tse_dir = os.path.join(self.save_dir, "tse_models")
        os.makedirs(self.tse_dir, exist_ok=True)

    def _get_checkpoint_path(self):
        return os.path.join(self.checkpoint_dir, f"ours_task_{self._cur_task}.pth")

    def _save_checkpoint(self, com_round, optimizer, scheduler):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "cur_task": self._cur_task,
            "com_round": com_round,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "model_state": self._network.state_dict(),
            "best_model": self.best_model,
            "best_val_auc": self.best_val_auc,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        torch.save(checkpoint, self._get_checkpoint_path())

    def _load_checkpoint(self, optimizer, scheduler):
        if int(self.args.get("resume", 1)) == 0:
            return None
        ckpt_path = self._get_checkpoint_path()
        if not os.path.exists(ckpt_path):
            return None
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if checkpoint.get("cur_task") != self._cur_task:
            return None
        self._network.load_state_dict(checkpoint["model_state"])
        self.best_model = checkpoint.get(
            "best_model", copy.deepcopy(self._network.state_dict())
        )
        self.best_val_auc = checkpoint.get("best_val_auc", 0)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        # if "scheduler_state" in checkpoint:
        #     scheduler.load_state_dict(checkpoint["scheduler_state"])
        return checkpoint

    def _get_tse_checkpoint_path(self, task_id):
        return os.path.join(self.tse_dir, f"tse_task_{task_id}.pth")

    def _build_tse_model(self, output_dim):
        tse_model = IncrementalNet(self.args, False, False)
        tse_model.update_fc(output_dim)
        return tse_model

    def _load_tse_model_for_task(self, task_id, device):
        ckpt_path = self._get_tse_checkpoint_path(task_id)
        if not os.path.exists(ckpt_path):
            raise ValueError(
                f"TSE checkpoint not found for task {task_id}: {ckpt_path}. "
                f"Please train preceding tasks first before using --syn_data_dir."
            )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        output_dim = int(ckpt["output_dim"])
        model = self._build_tse_model(output_dim)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        model.eval()
        return model, output_dim, ckpt

    def _local_update_tse(self, model, train_data_loader, lr, criterion, output_dim):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=float(
                self.args.get("tse_weight_decay", self.args["weight_decay"])
            ),
        )

        local_ep_cfg = int(self.args.get("tse_local_ep", self.args["local_ep"]))
        local_ep = local_ep_cfg if local_ep_cfg > 0 else int(self.args["local_ep"])
        total_loss = 0.0
        total_loss_f = 0.0
        for _ in range(local_ep):
            for images, labels in train_data_loader:
                images = images.cuda()
                labels = labels.cuda().float()
                output_list = model(images)
                logits = output_list["logits"][:, :output_dim]
                feature = output_list["att"]

                labels_cls = labels[:, :output_dim]
                label_tuples = [tuple(l.cpu().numpy()) for l in labels]
                target_f = torch.stack(
                    [self.label_embedding_cache[t].squeeze(0) for t in label_tuples],
                    dim=0,
                )
                target_f = target_f.to(feature.device)

                loss_f = torch.relu(
                    torch.nn.functional.mse_loss(feature, target_f.detach()) - self.r
                )

                loss_asl = criterion(logits, labels_cls)
                loss = loss_asl + self.tse_ltc * loss_f

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.detach().item()
                total_loss_f += loss_f.detach().item()

        return (
            model.state_dict(),
            total_loss,
            total_loss_f / (local_ep * len(train_data_loader)),
        )

    def _train_task_tse_federated(
        self, train_dataset, val_dataset, test_dataset, user_groups
    ):
        """
        Task-specific TSE federated training:
        - Train one TSE for current task with current-task data only.
        - Use Asymmetric Loss only.
        - Aggregate using the same federated averaging rule as the main model.
        - Keep global model with best validation mean AUC.
        """
        task_id = self._cur_task
        ckpt_path = self._get_tse_checkpoint_path(task_id)
        pretrained_tse_model_path = self.args.get(
            "pretrained_tse_model_path"
        ) or self.args.get("tse_model_path", "")

        if pretrained_tse_model_path and os.path.exists(pretrained_tse_model_path):
            # Resolve whether a directory with per-task ckpts was supplied
            if os.path.isdir(pretrained_tse_model_path):
                candidate = os.path.join(
                    pretrained_tse_model_path, f"tse_task_{task_id}.pth"
                )
            else:
                candidate = pretrained_tse_model_path

            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f"Pretrained TSE candidate not found: {candidate}"
                )

            # Load the checkpoint and assert output_dim matches expected
            ckpt = torch.load(candidate, map_location="cpu")
            if isinstance(ckpt, dict) and "output_dim" in ckpt:
                ckpt_output_dim = int(ckpt["output_dim"])
            else:
                # if no explicit output_dim saved, conservatively assume matches
                ckpt_output_dim = int(self._total_classes)

            assert ckpt_output_dim == int(self._total_classes), (
                f"Pretrained TSE output_dim ({ckpt_output_dim}) does not match expected "
                f"({self._total_classes}) for task {task_id}"
            )

            # If matches, materialize into standard tse checkpoint path and skip training
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                save_payload = copy.deepcopy(ckpt)
            else:
                save_payload = {
                    "task_id": task_id,
                    "output_dim": ckpt_output_dim,
                    "best_val_auc": (
                        float(ckpt.get("best_val_auc", 0.0))
                        if isinstance(ckpt, dict)
                        else 0.0
                    ),
                    "model_state": (
                        ckpt
                        if not isinstance(ckpt, dict)
                        else ckpt.get("model_state", ckpt)
                    ),
                }

            save_payload["task_id"] = task_id
            save_payload["output_dim"] = ckpt_output_dim
            torch.save(save_payload, ckpt_path)
            msg = (
                f"[TSE] task {task_id}: pretrained checkpoint provided, skip federated training: "
                f"{candidate} -> {ckpt_path}"
            )
            print(msg)
            self.logger.log(msg)
            return

        if int(self.args.get("resume", 0)) == 1 and os.path.exists(ckpt_path):
            msg = (
                f"[TSE] task {task_id}: checkpoint exists, skip retraining: {ckpt_path}"
            )
            print(msg)
            self.logger.log(msg)
            return

        output_dim = self._total_classes
        global_tse = self._build_tse_model(output_dim).cuda()

        tse_lr = float(self.args.get("tse_lr", self.args["local_lr"]))
        tse_frac = float(self.args.get("tse_frac", self.args["frac"]))

        tse_rounds = int(self.args["tse_com_rounds"][task_id])

        # criterion = AsymmetricLoss(
        #     gamma_neg=float(self.args.get("tse_gamma_neg", 4.0)),
        #     gamma_pos=float(self.args.get("tse_gamma_pos", 1.0)),
        #     clip=float(self.args.get("tse_clip", 0.05)),
        #     eps=float(self.args.get("tse_eps", 1e-8)),
        # )
        criterion = nn.BCEWithLogitsLoss()

        scheduler_optimizer = torch.optim.SGD(
            global_tse.parameters(),
            lr=tse_lr,
            momentum=0.9,
            weight_decay=float(
                self.args.get("tse_weight_decay", self.args["weight_decay"])
            ),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            scheduler_optimizer,
            T_max=max(tse_rounds, 1),
            eta_min=1e-3,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=int(self.args["num_worker"]),
            multiprocessing_context=self.args["mulc"],
            persistent_workers=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=int(self.args["num_worker"]),
            multiprocessing_context=self.args["mulc"],
            persistent_workers=True,
        )

        best_auc = -1.0
        best_state = copy.deepcopy(global_tse.state_dict())
        prog_bar = tqdm(range(tse_rounds), desc=f"TSE task {task_id}")

        for com in prog_bar:
            local_weights = []
            m = max(int(tse_frac * self.args["num_users"]), 1)
            idxs_users = np.random.choice(
                range(self.args["num_users"]), m, replace=False
            )

            for idx in idxs_users:
                tse_batch_cfg = int(
                    self.args.get("tse_batch_size", self.args["local_bs"])
                )
                tse_batch_size = (
                    tse_batch_cfg if tse_batch_cfg > 0 else int(self.args["local_bs"])
                )
                local_train_loader = DataLoader(
                    DatasetSplit(train_dataset, user_groups[idx]),
                    batch_size=tse_batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    persistent_workers=False,
                )
                local_model = copy.deepcopy(global_tse)
                w, _, loss_f = self._local_update_tse(
                    local_model,
                    local_train_loader,
                    scheduler.get_last_lr()[0],
                    criterion,
                    output_dim,
                )
                local_weights.append(copy.deepcopy(w))
                del local_model

            scheduler.step()

            global_weights = average_weights(local_weights)
            global_tse.load_state_dict(global_weights)

            _, val_auc = self._single_compute_auc(global_tse, val_loader)
            _, test_auc = self._single_compute_auc(global_tse, test_loader)
            if val_auc >= best_auc:
                best_auc = val_auc
                best_state = copy.deepcopy(global_tse.state_dict())

            round_msg = (
                f"[TSE] task {task_id}, round {com + 1}/{tse_rounds} => "
                f"val_auc={val_auc:.4f}, test_auc={test_auc:.4f}, loss_f={loss_f:.4f}"
            )
            self.logger.log(round_msg)

            prog_bar.set_description(
                f"TSE task {task_id} round {com + 1}/{tse_rounds} val_auc={val_auc:.4f} test_auc={test_auc:.4f} loss_f={loss_f:.4f}"
            )

        save_payload = {
            "task_id": task_id,
            "output_dim": output_dim,
            "best_val_auc": float(best_auc),
            "model_state": best_state,
        }
        torch.save(save_payload, ckpt_path)
        _, test_auc = self._single_compute_auc(global_tse, test_loader)
        msg = f"[TSE] task {task_id} trained. best_val_auc={best_auc:.4f}, test_auc={test_auc:.4f}, saved={ckpt_path}"
        print(msg)
        self.logger.log(msg)

        del global_tse
        torch.cuda.empty_cache()

    def calculate_LTE(self, train_dataset):
        # Collect all labels.
        all_labels = train_dataset.muti_labels
        if isinstance(all_labels, torch.Tensor):
            all_labels = all_labels.cpu().numpy()

        # Build unique label combinations.
        unique_label_tuples = list(set(tuple(l) for l in all_labels))

        # Compute CLIP embeddings in batches.
        for label_tuple in unique_label_tuples:
            # Convert tuples to tensors before encoding.
            label_tensor = torch.tensor(label_tuple, dtype=torch.float32).unsqueeze(0)
            emb = self.clip_model.caculate_embdding(label_tensor)  # (1, emb_dim)
            self.label_embedding_cache[label_tuple] = emb.detach().cpu()

        print(f"Precomputed {len(unique_label_tuples)} unique label embeddings.")

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        _, test_auc = self._compute_accuracy(self._old_network, self.test_loader)
        print("After Test AUC: %s" % test_auc)
        self.logger.log("After Test AUC: %s" % test_auc)
        print("\n✅ Task %d completed!\n" % self._cur_task)

    def _get_current_task_label_distribution(self, data_manager, total_budget=1000):
        current_datasets = data_manager.get_dataset(
            begin=self._cur_task, end=self._cur_task, mode="train"
        )

        if len(current_datasets) == 0:
            return []

        all_labels = []
        for dataset in current_datasets:
            if hasattr(dataset, "muti_labels"):
                labels = dataset.muti_labels
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                all_labels.extend(labels)

        if len(all_labels) == 0:
            return []

        # Count occurrences of each label combination.
        from collections import Counter

        label_counts = Counter(tuple(l) for l in all_labels)
        total_real_samples = sum(label_counts.values())
        if total_real_samples == 0:
            return []

        sampled_labels = []
        label_tuples = list(label_counts.keys())
        per_combo = total_budget // len(label_tuples)
        remainder = total_budget % len(label_tuples)
        for i, label_tuple in enumerate(label_tuples):
            cur_quota = per_combo + (1 if i < remainder else 0)
            for _ in range(cur_quota):
                sampled_labels.append(torch.tensor(label_tuple, dtype=torch.float32))

        print(
            f"Task {self._cur_task}: {len(label_counts)} unique label combinations, {total_real_samples} real samples"
        )
        print(
            f"Generation budget={total_budget}; sampled {len(sampled_labels)} label vectors by frequency"
        )
        return sampled_labels

    def _score_tse_images(self, tse_model, images, labels, device):
        dataset = InMemoryLabeledImageDataset(
            images, labels, transform=get_eval_transform()
        )
        loader = DataLoader(
            dataset,
            batch_size=int(self.args.get("tse_filter_bs", 128)),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )
        records = []
        tse_model.eval()
        with torch.no_grad():
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.float()
                output = tse_model(batch_images)
                logits = output["logits"]
                features = output["att"]
                label_tuples = [
                    tuple(int(v) for v in label.cpu().numpy().tolist())
                    for label in batch_labels
                ]
                targets = []
                for label_tuple in label_tuples:
                    if label_tuple not in self.label_embedding_cache:
                        label_tensor = torch.tensor(
                            label_tuple, dtype=torch.float32
                        ).unsqueeze(0)
                        embedding = self.clip_model.caculate_embdding(label_tensor)
                        self.label_embedding_cache[label_tuple] = (
                            embedding.detach().cpu()
                        )
                    targets.append(self.label_embedding_cache[label_tuple].squeeze(0))
                targets = torch.stack(targets, dim=0).to(features.device)
                if features.shape[1] != targets.shape[1]:
                    raise ValueError(
                        "TSE feature and LTE embedding dimensions do not match: "
                        f"{features.shape[1]} != {targets.shape[1]}"
                    )
                mse = torch.mean((features - targets) ** 2, dim=1)
                uncertainty = 0.5 - torch.mean(
                    torch.abs(torch.sigmoid(logits) - 0.5), dim=1
                )
                for idx, label_tuple in enumerate(label_tuples):
                    records.append(
                        {
                            "index": len(records),
                            "label": label_tuple,
                            "mse": float(mse[idx].cpu()),
                            "uncertainty": float(uncertainty[idx].cpu()),
                        }
                    )
        return records

    def _save_selected_synthetic_images(
        self, task_dir, selected_images, selected_labels
    ):
        os.makedirs(task_dir, exist_ok=True)
        for filename in os.listdir(task_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".npy")):
                os.remove(os.path.join(task_dir, filename))
        label_vectors = []
        for idx, (image, label) in enumerate(zip(selected_images, selected_labels)):
            image.save(os.path.join(task_dir, f"{idx:06d}.png"))
            label_vectors.append(np.asarray(label, dtype=np.float32))
        np.save(os.path.join(task_dir, "labels.npy"), np.asarray(label_vectors))

    def data_generation_with_minim(self, minim_model_path, data_manager):
        """
        Generate synthetic data with MINIM from the current task distribution.

        Args:
            minim_model_path: MINIM model path.
            data_manager: Dataset manager.
        """
        print(
            f"Generating MINIM synthetic data for task {self._cur_task} from the current task distribution..."
        )

        # Get the current task label distribution.
        current_task_labels = self._get_current_task_label_distribution(
            data_manager, total_budget=self.args.get("pre_nums")
        )

        if len(current_task_labels) == 0:
            print("No current-task data; skipping generation.")
            return

        # Initialize the MINIM generator.
        minim_generator = MINIMGenerator(
            minim_model_path=minim_model_path,
            pretrained_model_path=self.args["minim_pretrained_model_path"],
            device=f"cuda:{self.args['gpu']}",
            num_inference_steps=self.args.get("minim_num_inference_steps"),
            guidance_scale=self.args.get("minim_guidance_scale"),
            clip_similarity_threshold=self.args.get(
                "minim_clip_similarity_threshold", 0
            ),
            max_regen_attempts=self.args.get("minim_max_regen_attempts", 200),
        )

        # Frequency allocation already repeats current_task_labels; no fixed per-class count is needed.
        # Use total_budget as the upstream allocation and record the total to generate.
        expected_image_count = len(current_task_labels)
        print(
            f"Preparing to generate images: Given {len(current_task_labels)} frequency-assigned label vectors, approximately {expected_image_count} images will be generated."
        )

        # Generate images under args["save_dir"].
        save_dir = self.args.get("save_dir", self.save_dir)
        tmp_dir = os.path.join(save_dir, "task_{}".format(self._cur_task))
        all_images, all_labels, _ = minim_generator.generate_from_labels(
            current_task_labels,
            self.clip_model,
            batch_size=self.args["minim_generation_batch_size"],
        )

        label_tuples = [
            tuple(int(v) for v in label.cpu().numpy().tolist())
            for label in current_task_labels
        ]
        target_counts = defaultdict(int)
        for label_tuple in label_tuples:
            target_counts[label_tuple] += 1
        combos = sorted(target_counts)
        target_nums = int(self.args.get("nums", self.nums))
        if target_nums > len(current_task_labels):
            raise ValueError(
                f"nums ({target_nums}) cannot exceed pre_nums ({len(current_task_labels)})"
            )

        device = f"cuda:{self.args['gpu']}"
        tse_model, _, _ = self._load_tse_model_for_task(self._cur_task, device)
        filter_radius = float(self.args.get("tse_filter_radius", 0.015))
        max_rounds = int(self.args.get("minim_filter_max_rounds", 5))
        accepted_records = None

        for supplement_round in range(max_rounds + 1):
            records = self._score_tse_images(tse_model, all_images, all_labels, device)
            accepted = defaultdict(list)
            for record in records:
                if record["mse"] <= filter_radius:
                    accepted[record["label"]].append(record)

            missing_labels = []
            for combo in combos:
                accepted_count = min(len(accepted[combo]), target_counts[combo])
                missing_labels.extend(
                    [torch.tensor(combo, dtype=torch.float32)]
                    * (target_counts[combo] - accepted_count)
                )

            if not missing_labels:
                accepted_records = accepted
                break
            if supplement_round >= max_rounds:
                missing = {
                    combo: target_counts[combo]
                    - min(len(accepted[combo]), target_counts[combo])
                    for combo in combos
                    if len(accepted[combo]) < target_counts[combo]
                }
                raise RuntimeError(
                    "TSE-LTE filtering could not satisfy all label combinations "
                    f"within {max_rounds} supplement rounds: {missing}"
                )

            extra_images, extra_labels, _ = minim_generator.generate_from_labels(
                missing_labels,
                self.clip_model,
                batch_size=self.args["minim_generation_batch_size"],
            )
            all_images.extend(extra_images)
            all_labels.extend(extra_labels)
            print(
                f"TSE supplement generation round {supplement_round + 1}: "
                f"added {len(extra_images)} images"
            )

        first_stage_images = []
        first_stage_labels = []
        for combo in combos:
            selected = sorted(accepted_records[combo], key=lambda item: item["mse"])
            for record in selected[: target_counts[combo]]:
                first_stage_images.append(all_images[record["index"]])
                first_stage_labels.append(all_labels[record["index"]])

        final_counts = {}
        per_combo = target_nums // len(combos)
        remainder = target_nums % len(combos)
        for idx, combo in enumerate(combos):
            final_counts[combo] = per_combo + (1 if idx < remainder else 0)

        final_images = []
        final_labels = []
        first_records = self._score_tse_images(
            tse_model, first_stage_images, first_stage_labels, device
        )
        for combo in combos:
            selected = sorted(
                [record for record in first_records if record["label"] == combo],
                key=lambda item: item["uncertainty"],
                reverse=True,
            )
            if len(selected) < final_counts[combo]:
                raise RuntimeError(
                    f"Not enough first-stage samples for label combination {combo}: "
                    f"{len(selected)} < {final_counts[combo]}"
                )
            for record in selected[: final_counts[combo]]:
                final_images.append(first_stage_images[record["index"]])
                final_labels.append(first_stage_labels[record["index"]])

        self._save_selected_synthetic_images(tmp_dir, final_images, final_labels)
        print(
            f"MINIM two-stage filtering complete: stage one {len(first_stage_images)} images, "
            f"saved {len(final_images)} images to {tmp_dir}"
        )

        # Release the generator to free GPU memory.
        del minim_generator
        del tse_model
        torch.cuda.empty_cache()

    def get_syn_data_loader(self, data_manager, base_dir=None):
        """
        Replay synthetic data generated by MINIM for previous tasks.
        """
        syn_bs = self.args["syn_bs"] * self.args["local_bs"]

        # Load generated images for tasks 0 through t-1.
        # Use the directory specified by args["save_dir"].
        save_dir = base_dir or self.args.get("save_dir", self.save_dir)
        syn_datasets = []
        total_images = 0

        for task_idx in range(self._cur_task):
            data_dir = os.path.join(save_dir, "task_{}".format(task_idx))
            if os.path.exists(data_dir):
                syn_dataset = LabeledImageDataset(
                    data_dir,
                    transform=self.transform,
                    nums=self.nums,
                    sample_seed=int(self.select_img_seed),
                )

                dataset_size = len(syn_dataset)
                if dataset_size > 0:
                    syn_datasets.append(syn_dataset)
                    total_images += dataset_size
                    print(
                        f"Load the generated data for task {task_idx}: {dataset_size} images"
                    )

        if len(syn_datasets) == 0:
            print(
                "Warning: No MINIM generated data found for replay. Skip synthetic replay this round."
            )
            return None
        else:
            print(
                f"A total of {len(syn_datasets)} tasks' generated data were loaded, resulting in {total_images} images for playback."
            )

        syn_dataset = ConcatDataset(syn_datasets)

        # Custom collate aligns label lengths to the current output dimension.
        out_dim = self._total_classes

        def syn_collate_fn(batch):
            if isinstance(batch[0], tuple):
                imgs, labels, mask = zip(*batch)
                imgs = torch.stack(imgs, dim=0)
                labels_t = []
                for lab in labels:
                    lab = lab.float()
                    if lab.shape[0] < out_dim:
                        pad_cols = out_dim - lab.shape[0]
                        lab = F.pad(lab, (0, pad_cols), mode="constant", value=0.0)
                    elif lab.shape[0] > out_dim:
                        lab = lab[:out_dim]
                    labels_t.append(lab)
                labels = torch.stack(labels_t, dim=0)
                return imgs, labels, mask
            else:
                return torch.stack(batch, dim=0)

        syn_num_workers = int(self.args["num_worker"])
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset,
            batch_size=syn_bs,
            shuffle=True,
            num_workers=syn_num_workers,
            multiprocessing_context=self.args["mulc"],
            persistent_workers=True,
            collate_fn=syn_collate_fn,
        )
        return syn_data_loader

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_dataset = self.dataset_names[self._cur_task]
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self.transform = get_norm_and_transform()

        self._network.update_fc(self._total_classes)
        message = "Learning on {}: {}-{}".format(
            self._cur_dataset, self._known_classes, self._total_classes
        )
        print(message)
        self.logger.log(message)

        if self.args["log_loss"]:
            self.logger.log_loss(
                "###################################################################################################################"
            )
            self.logger.log_loss(
                f"Start training on task {self._cur_task} ({self._cur_dataset}) with classes {self._known_classes} to {self._total_classes - 1}"
            )

        # print(self._cur_task)
        train_dataset = data_manager.get_dataset(  # * get the data for one task
            begin=self._cur_task, end=self._cur_task, mode="train"
        )
        val_datasets = data_manager.get_dataset(begin=0, end=self._cur_task, mode="val")
        test_datasets = data_manager.get_dataset(
            begin=0, end=self._cur_task, mode="test"
        )

        self.calculate_LTE(train_dataset[0])

        self.val_loader = []
        for val_dataset in val_datasets:
            val_num_workers = int(self.args["num_worker"])
            val_loader = DataLoader(
                val_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=val_num_workers,
                multiprocessing_context=self.args["mulc"],
                persistent_workers=True,
            )
            self.val_loader.append(val_loader)

        self.test_loader = []
        for test_dataset in test_datasets:
            test_num_workers = int(self.args["num_worker"])
            test_loader = DataLoader(
                test_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=test_num_workers,
                multiprocessing_context=self.args["mulc"],
                persistent_workers=True,
            )
            self.test_loader.append(test_loader)

        if self._cur_task > 0:
            old_datasets = data_manager.get_dataset(
                begin=0, end=self._cur_task - 1, mode="test"
            )
            self.old_loader = []
            for old_dataset in old_datasets:
                old_num_workers = int(self.args["num_worker"])
                old_loader = DataLoader(
                    old_dataset,
                    batch_size=256,
                    shuffle=False,
                    num_workers=old_num_workers,
                    multiprocessing_context=self.args["mulc"],
                    persistent_workers=True,
                )
                self.old_loader.append(old_loader)

            new_dataset = data_manager.get_dataset(
                begin=self._cur_task, end=self._cur_task, mode="test"
            )
            new_num_workers = int(self.args["num_worker"])
            self.new_loader = DataLoader(
                new_dataset[0],
                batch_size=256,
                shuffle=False,
                num_workers=new_num_workers,
                multiprocessing_context=self.args["mulc"],
                persistent_workers=True,
            )

        setup_seed(self.seed)
        if self._cur_task == 0 and (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)

        # 1. Federated training.
        print(f"\n📌 Start federated training for task {self._cur_task}...\n")

        user_groups, _ = partition_data(
            train_dataset[0].muti_labels,
            beta=self.args["beta"],
            n_parties=self.args["num_users"],
        )
        self.current_task_train_dataset = train_dataset[0]
        self.current_user_groups = user_groups

        # For later tasks, preload historical synthetic data for replay.
        external_syn_dir = self.args.get("syn_data_dir")
        if self._cur_task > 0:
            self.syn_data_loader = self.get_syn_data_loader(
                data_manager, base_dir=external_syn_dir
            )
            if self.syn_data_loader is None:
                print(
                    "Info: No synthetic replay data available. Finetune will run without replay for this round."
                )

        completed = self._fl_train(
            user_groups, train_dataset[0], self.val_loader, self.test_loader
        )  # train_dataset is a list

        # 2. Task-level TSE training and federated aggregation.
        if self._cur_task < self.tasks - 1 and self.args.get("use_tse_filter", True):
            current_val_dataset = val_datasets[-1]
            current_test_dataset = test_datasets[-1]
            print(
                f"\n📌 Start task-specific TSE federated training for task {self._cur_task}...\n"
            )
            self._train_task_tse_federated(
                train_dataset=train_dataset[0],
                val_dataset=current_val_dataset,
                test_dataset=current_test_dataset,
                user_groups=user_groups,
            )

        # Skip generation and filtering when pre-generated images are supplied.
        if completed:
            minim_model_dir = os.path.join(
                self.save_dir, f"minim_task_{self._cur_task}", "unet"
            )
            syn_base = external_syn_dir or self.args.get("save_dir", self.save_dir)
            syn_dir = os.path.join(syn_base, f"task_{self._cur_task}")
            syn_labels = os.path.join(syn_dir, "labels.npy")
            minim_ready = os.path.exists(minim_model_dir)
            syn_ready = os.path.exists(syn_labels)
            if minim_ready and syn_ready:
                msg = f"Task {self._cur_task} has been completed (breakpoint resumed), skipping MINIM training and synthetic data generation."
                print(msg)
                self.logger.log(msg)
                return

        # 3. Federated MINIM training.
        if (
            self._cur_task < self.tasks - 1
            and self.args.get("use_minim", True)
            and not external_syn_dir
        ):
            print(
                f"\n📌 Start federated training of the MINIM model for task {self._cur_task}...\n"
            )

            minim_trainer = MINIMTrainer(self.args, clip_model=self.clip_model)

            # Train the MINIM model with federated clients.
            minim_model_path = minim_trainer.train_minim_federated(
                train_dataset=train_dataset[0],
                output_dir=self.save_dir,
                task_id=self._cur_task,
                user_groups=user_groups,
            )

            # 4. Generate and filter synthetic data with the aggregated model.
            print(
                f"📌 Start using the aggregated MINIM model to generate synthetic data for task {self._cur_task}...\n"
            )
            self.data_generation_with_minim(minim_model_path, data_manager)

            # Release the trainer to free GPU memory.
            del minim_trainer
            torch.cuda.empty_cache()

    def _fl_train(self, user_groups, train_dataset, val_loader, test_loader):
        self._network.cuda()
        self.best_model = copy.deepcopy(
            self._network.state_dict()
        )  # Best model using the lowest training loss
        self.best_val_auc = 0

        total_rounds = self.args["com_rounds"][self._cur_task]
        warmup_rounds = self.args.get(
            "warmup_rounds", int(0.1 * total_rounds)
        )  # Warm up for 10% of the rounds by default.

        # Define a linearly increasing warmup schedule.
        def warmup_lambda(round):
            if round < warmup_rounds:
                return (round + 1) / warmup_rounds  # From 0 to 1.
            else:
                return 1.0

        optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self.args["local_lr"],
            betas=(0.9, 0.999),
            weight_decay=self.args["weight_decay"],
        )

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_rounds - warmup_rounds,
            eta_min=self.args["local_lr"] * 0.1,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_rounds],
        )

        start_round = 0
        checkpoint = self._load_checkpoint(optimizer, scheduler)
        if checkpoint is not None:
            start_round = int(checkpoint.get("com_round", 0))
            print(
                f"Resuming from checkpoint: task {self._cur_task}, round {start_round}/{self.args['com_rounds'][self._cur_task]}"
            )
            self.logger.log(
                f"Resume checkpoint: task {self._cur_task}, com_round {start_round}/{self.args['com_rounds'][self._cur_task]}"
            )

        if start_round >= self.args["com_rounds"][self._cur_task]:
            self._network.load_state_dict(self.best_model)
            torch.cuda.empty_cache()
            return True

        prog_bar = tqdm(range(start_round, self.args["com_rounds"][self._cur_task]))
        for com in prog_bar:
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(
                range(self.args["num_users"]), m, replace=False
            )
            loss_weight = []
            for idx in idxs_users:
                local_train_loader = DataLoader(
                    DatasetSplit(train_dataset, user_groups[idx]),
                    batch_size=self.args["local_bs"],
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    # multiprocessing_context=self.args["mulc"],
                    persistent_workers=False,
                )
                if self._cur_task == 0:
                    w, total_loss = self._local_update(
                        copy.deepcopy(self._network),
                        local_train_loader,
                        scheduler.get_last_lr()[0],
                    )
                else:
                    w, total_syn, total_local, total_loss = self._local_finetune(
                        self._old_network,
                        copy.deepcopy(self._network),
                        local_train_loader,
                        self._cur_task,
                        idx,
                        scheduler.get_last_lr()[0],
                    )
                    if com == 0 and self._cur_task == 1:
                        print(
                            "\t \t client {}, local dataset size:{},  syntheic data size:{}".format(
                                idx, total_local, total_syn
                            )
                        )
                local_weights.append(copy.deepcopy(w))
                loss_weight.append(total_loss)
                del local_train_loader, w
                # torch.cuda.empty_cache()
            scheduler.step()

            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            val_auc = self._compute_auc(copy.deepcopy(self._network), val_loader)
            if val_auc >= self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_model = copy.deepcopy(self._network.state_dict())

            if com % 1 == 0 and com < self.args["com_rounds"][self._cur_task]:
                test_auc_per = self._compute_auc(self._network, test_loader)
                if self._cur_task > 0:
                    test_old_auc = self._compute_auc(
                        copy.deepcopy(self._network), self.old_loader
                    )
                    test_new_auc = self._compute_auc(
                        copy.deepcopy(self._network), self.new_loader
                    )
                    print(
                        "Task {}, Test_AUC {} Old {} New {}\n".format(
                            self._cur_task, test_auc_per, test_old_auc, test_new_auc
                        )
                    )
                    self.logger.log(
                        "Task {}, Test_AUC {} Old {} New {}".format(
                            self._cur_task, test_auc_per, test_old_auc, test_new_auc
                        )
                    )

                info = "Task {}, Epoch {}/{} =>  Val_AUC {}, Test_AUC {}\n".format(
                    self._cur_task,
                    com + 1,
                    self.args["com_rounds"][self._cur_task],
                    val_auc,
                    test_auc_per,
                )
                print(info)
                self.logger.log(info)
                prog_bar.set_description(info)

            self._save_checkpoint(com + 1, optimizer, scheduler)
        self._network.load_state_dict(
            self.best_model
        )  # Best model using the lowest training loss
        del self.best_model
        torch.cuda.empty_cache()
        return False

    def _local_update(self, model, train_data_loader, lr):
        print(f"lr = {lr}")
        model.train()
        total_loss = 0
        total_ce_loss = 0
        total_f_loss = 0
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=self.args["weight_decay"],
        )
        for iter in range(self.args["local_ep"]):
            for batch_idx, (images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                # print(images.shape)
                output_list = model(images)
                output = output_list["logits"]
                feature = output_list["att"]

                label_tuples = [tuple(l.cpu().numpy()) for l in labels]
                target_f = torch.stack(
                    [self.label_embedding_cache[t].squeeze(0) for t in label_tuples],
                    dim=0,
                )
                target_f = target_f.to(feature.device)

                if self.args.get("use_semantic", 0) == 1:
                    loss_f = torch.relu(
                        torch.nn.functional.mse_loss(feature, target_f.detach())
                        - self.r
                    )
                else:
                    loss_f = torch.tensor(0.0).to(feature.device)

                loss_ce = F.binary_cross_entropy_with_logits(output, labels)
                loss = loss_ce + self.ltc * loss_f
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter == 0:
                    total_loss += loss.detach()
                    total_ce_loss += loss_ce.detach()
                    total_f_loss += loss_f.detach()
        print(
            "---task {} =>  CE: {} F: {} | CE: {}, F: {} T: {}".format(
                self._cur_task,
                loss_ce.detach(),
                loss_f.detach(),
                total_ce_loss,
                total_f_loss,
                total_loss,
            )
        )

        return model.state_dict(), total_loss

    def _local_finetune(
        self, teacher, model, train_data_loader, task_id, client_id, lr
    ):
        alpha = np.log2(self._total_classes / 2 + 1)
        beta = np.sqrt(self._known_classes / self._total_classes)
        cur = self.args["cur"] * (1 + 1 / alpha) / beta
        pre = self.args["pre"] * alpha * beta

        model.train()
        teacher.eval()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=self.args["weight_decay"],
        )
        syn_data_iter = DataIter(self.syn_data_loader)

        # Define the attention-transfer function explicitly.
        def get_attention_map(feature_map):
            # For (B, C, H, W), aggregate channel magnitudes and apply L2 normalization.
            # This captures the teacher's spatial attention manifold.
            am = torch.pow(feature_map, 2).mean(dim=1)  # (B, H, W)
            am = am.view(am.size(0), -1)  # (B, H*W)
            am = torch.nn.functional.normalize(am, p=2, dim=1)
            return am

        for it in range(self.args["local_ep"]):
            iter_loader = enumerate(train_data_loader)
            total_local = 0.0
            total_syn = 0.0
            total_ce_loss = 0
            total_f_loss = 0
            total_kd_loss = 0
            total_sf_loss = 0

            total_mid_feat_loss = 0
            total_attn_loss = 0

            total_syn_cross_loss = 0
            total_loss = 0

            for batch_idx, (images, labels) in iter_loader:
                images, labels = images.cuda(), labels.cuda()

                c_output_list = model(images)
                c_output = c_output_list["logits"]
                c_feature = c_output_list["att"]

                label_tuples = [tuple(l.cpu().numpy()) for l in labels]
                target_f = torch.stack(
                    [self.label_embedding_cache[t].squeeze(0) for t in label_tuples],
                    dim=0,
                )
                c_target_f = target_f.to(c_feature.device)

                if self.args.get("use_semantic", 0) == 1:
                    c_loss_f = torch.relu(
                        torch.nn.functional.mse_loss(c_feature, c_target_f.detach())
                        - self.r
                    )
                else:
                    c_loss_f = torch.tensor(0.0).to(c_feature.device)

                loss_ce_cur = F.binary_cross_entropy_with_logits(c_output, labels)

                total_local += images.shape[0]

                if syn_data_iter is not None:
                    syn_input, _, mask = syn_data_iter.next()
                    syn_input = syn_input.cuda()

                    s_out_list = model(syn_input.detach())
                    s_out = s_out_list["logits"]
                    s_f = s_out_list["att"]

                    # --- Extract intermediate student feature maps. ---
                    s_m1 = s_out_list.get("feat_m1", None)
                    s_m2 = s_out_list.get("feat_m2", None)

                    with torch.no_grad():
                        t_out_list = teacher(syn_input.detach())
                        t_out = t_out_list["logits"]
                        t_f = t_out_list["att"]

                        # --- Extract intermediate teacher feature maps. ---
                        t_m1 = t_out_list.get("feat_m1", None)
                        t_m2 = t_out_list.get("feat_m2", None)

                        total_syn += syn_input.shape[0]

                    # 1. Original logit KD.
                    loss_kd = _KD_loss(
                        s_out[:, : self._known_classes],
                        t_out.detach(),
                        tau,
                    )

                    # 2. Final feature MSE.
                    if self.args.get("use_semantic", 0) == 1:
                        s_loss_f = torch.nn.functional.mse_loss(s_f, t_f.detach())
                    else:
                        s_loss_f = torch.tensor(0.0).to(s_f.device)

                    # 3. Intermediate feature-manifold MSE.
                    loss_mid_feat = 0.0
                    if s_m1 is not None and t_m1 is not None:
                        loss_mid_feat += torch.nn.functional.mse_loss(
                            s_m1, t_m1.detach()
                        )
                    if s_m2 is not None and t_m2 is not None:
                        loss_mid_feat += torch.nn.functional.mse_loss(
                            s_m2, t_m2.detach()
                        )

                    # 4. Spatial attention-transfer loss.
                    loss_attn = 0.0
                    if s_m1 is not None and t_m1 is not None:
                        loss_attn += torch.nn.functional.mse_loss(
                            get_attention_map(s_m1), get_attention_map(t_m1.detach())
                        )
                    if s_m2 is not None and t_m2 is not None:
                        loss_attn += torch.nn.functional.mse_loss(
                            get_attention_map(s_m2), get_attention_map(t_m2.detach())
                        )

                    # --- Recombine constraints for previously learned knowledge. ---
                    # Use mu_feat and mu_attn to balance feature-loss magnitudes.
                    mu_feat = self.args.get("mu_feat", 0.1)
                    mu_attn = self.args.get("mu_attn", 0.5)

                    total_pre_loss = (
                        loss_kd
                        + s_loss_f
                        + (mu_feat * loss_mid_feat)
                        + (mu_attn * loss_attn)
                    )

                    loss = (
                        cur * (loss_ce_cur + self.ltc * c_loss_f) + pre * total_pre_loss
                    )
                else:
                    raise ValueError("without syn_dataset")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.detach() * images.shape[0]
                total_ce_loss += loss_ce_cur.detach() * images.shape[0]
                total_f_loss += c_loss_f.detach() * images.shape[0]
                total_kd_loss += loss_kd.detach() * images.shape[0]
                total_sf_loss += s_loss_f.detach() * images.shape[0]

                # Record the additional loss terms.
                if s_m1 is not None:
                    total_mid_feat_loss += loss_mid_feat.detach() * images.shape[0]
                    total_attn_loss += loss_attn.detach() * images.shape[0]

            print(
                "🏥 Client {}, ep {}/{}, Loss: {:.4f} => CE: {:.4f}, CF: {:.4f} || KL: {:.4f}, SF: {:.4f}, MidFeat: {:.4f}, Attn: {:.4f}".format(
                    client_id,
                    it + 1,
                    self.args["local_ep"],
                    total_loss / total_local,
                    total_ce_loss / total_local,
                    total_f_loss / total_local,
                    total_kd_loss / total_local,
                    total_sf_loss / total_local,
                    total_mid_feat_loss / total_local,
                    total_attn_loss / total_local,
                )
            )
            if self.args["log_loss"]:
                self.logger.log_loss(
                    "Client {}, ep {}/{}, Loss: {:.6f} => CE: {:.6f}, CF: {:.6f} || KL: {:.6f}, SF: {:.6f}, MidFeat: {:.6f}, Attn: {:.6f}".format(
                        client_id,
                        it + 1,
                        self.args["local_ep"],
                        total_loss / total_local,
                        total_ce_loss / total_local,
                        total_f_loss / total_local,
                        total_kd_loss / total_local,
                        total_sf_loss / total_local,
                        total_mid_feat_loss / total_local,
                        total_attn_loss / total_local,
                    )
                )
                tloss_cur = (
                    total_ce_loss / total_local + self.ltc * total_f_loss / total_local
                )
                tloss_pre = (
                    total_kd_loss / total_local
                    + total_sf_loss / total_local
                    + (mu_feat * total_mid_feat_loss / total_local)
                    + (mu_attn * total_attn_loss / total_local)
                )
                self.logger.log_loss(
                    "Cur loss: {:.6f}, Pre loss: {:.6f}, Total loss: {:.6f} || cur {}, pre {}".format(
                        tloss_cur, tloss_pre, total_loss / total_local, cur, pre
                    )
                )

        return model.state_dict(), total_syn, total_local, total_loss


def _KD_loss(pred, soft, T):
    return F.binary_cross_entropy_with_logits(pred / T, torch.sigmoid(soft / T)) * (
        T**2
    )
