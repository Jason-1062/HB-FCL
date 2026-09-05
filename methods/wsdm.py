from copy import deepcopy
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np
import skfuzzy as fuzz
import torch
import torch.optim as optim
from PIL import Image
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from methods.base import BaseLearner
from utils.Scheduler import GradualWarmupScheduler
from utils.WSDM_Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from utils.WSDM_Model import UNet
from utils.data_manager import average_weights, partition_data, setup_seed
from utils.inc_net import IncrementalNet

wsdm_com_rounds = 10
wsdm_local_ep = 2


def build_syn_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_diffusion_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )


class IndexedDatasetSplit(Dataset):
    def __init__(self, dataset: Dataset, idxs: List[int]):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int):
        idx = self.idxs[item]
        image, label = self.dataset[idx]
        return idx, image, label


class DiffusionIndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, idxs: List[int], transform):
        self.base_dataset = base_dataset
        self.idxs = [int(i) for i in idxs]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int):
        idx = self.idxs[item]
        image_path = self.base_dataset.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.base_dataset.muti_labels[idx]
        return idx, image, label


class WSDMSynDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, target_dim: int = None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_dim = target_dim
        self.paths: List[str] = []
        self.labels: List[np.ndarray] = []
        if not os.path.isdir(root_dir):
            return

        meta_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.startswith("metadata_client") and f.endswith(".npz")
        ]
        for meta_file in sorted(meta_files):
            data = np.load(meta_file, allow_pickle=True)
            rel_paths = data["paths"]
            labels = data["labels"]
            for rel_path, label in zip(rel_paths.tolist(), labels):
                abs_path = os.path.join(root_dir, rel_path)
                if os.path.isfile(abs_path):
                    self.paths.append(abs_path)
                    self.labels.append(np.asarray(label, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_dim is not None:
            if label.shape[0] < self.target_dim:
                label = np.pad(label, (0, self.target_dim - label.shape[0]), mode="constant")
            elif label.shape[0] > self.target_dim:
                label = label[: self.target_dim]
        label = torch.from_numpy(label.astype(np.float32))
        return image, label


class WSDM(BaseLearner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self._network = IncrementalNet(args, False)

        self.cluster_num = int(args.get("wsdm_clusters", 2))
        self.fuzz_m = float(args.get("wsdm_m", 1.1))
        self.diffusion_steps = int(args.get("wsdm_T", 1000))
        self.g_batch_size = int(args.get("wsdm_gen_batch_size", 7))
        self.mid_round_ratio = float(args.get("wsdm_mid_round_ratio", 0.5))
        self.diffusion_image_size = int(args.get("wsdm_image_size", 256))
        self.wsdm_train_batch_size = int(args.get("wsdm_train_batch_size", 1))
        self.wsdm_amp = bool(int(args.get("wsdm_amp", 1)))

        default_save_path = os.path.join(self.save_dir, "wsdm_data")
        self.wsdm_save_path = args.get("wsdm_save_path", default_save_path) or default_save_path

        self.syn_transform = build_syn_transform()
        self.diffusion_transform = build_diffusion_transform(self.diffusion_image_size)

        self.combo_to_id: Dict[Tuple[int, ...], int] = {}
        self.id_to_combo: Dict[int, Tuple[int, ...]] = {}
        self.combo_id_to_multihot: Dict[int, np.ndarray] = {}
        self.sample_idx_to_combo: Dict[int, int] = {}
        self.combo_count = 0

        self.local_train_loaders: List[DataLoader] = []
        self.val_loader: List[DataLoader] = []
        self.test_loader: List[DataLoader] = []
        self.fl_syn_train_loader = None
        self.cl_syn_train_loader = None

    def _loader_kwargs(self, shuffle: bool) -> Dict:
        num_workers = int(self.args.get("num_worker", 4))
        kwargs = {
            "batch_size": self.args["local_bs"],
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            mulc = self.args.get("mulc", None)
            if mulc is not None:
                kwargs["multiprocessing_context"] = mulc
        return kwargs

    def _build_eval_loader(self, dataset: Dataset) -> DataLoader:
        num_workers = int(self.args.get("num_worker", 4))
        kwargs = {
            "batch_size": 256,
            "shuffle": False,
            "num_workers": num_workers,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            mulc = self.args.get("mulc", None)
            if mulc is not None:
                kwargs["multiprocessing_context"] = mulc
        return DataLoader(dataset, **kwargs)

    def _dataset_labels_as_multihot(self, dataset: Dataset) -> np.ndarray:
        labels = getattr(dataset, "muti_labels", None)
        if labels is None:
            labels = getattr(dataset, "labels", None)
        if labels is None:
            raise ValueError("Dataset has neither 'muti_labels' nor 'labels'.")

        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels)

        if labels.ndim == 1:
            num_classes = max(int(labels.max()) + 1, self._total_classes if self._total_classes > 0 else 1)
            labels = np.eye(num_classes, dtype=np.float32)[labels.astype(np.int64)]
        return labels.astype(np.float32)

    @staticmethod
    def _multihot_to_key(label_vec: np.ndarray) -> Tuple[int, ...]:
        pos = np.where(label_vec > 0.5)[0]
        if len(pos) == 0:
            return (-1,)
        return tuple(pos.tolist())

    def _build_combo_mapping(self, train_set: Dataset):
        labels = self._dataset_labels_as_multihot(train_set)

        key_to_vec: Dict[Tuple[int, ...], np.ndarray] = {}
        for label_vec in labels:
            key = self._multihot_to_key(label_vec)
            if key not in key_to_vec:
                key_to_vec[key] = label_vec.copy()

        keys = sorted(key_to_vec.keys())
        self.combo_to_id = {k: i for i, k in enumerate(keys)}
        self.id_to_combo = {v: k for k, v in self.combo_to_id.items()}
        self.combo_count = len(self.combo_to_id)

        self.combo_id_to_multihot = {}
        for key, combo_id in self.combo_to_id.items():
            vec = key_to_vec[key].astype(np.float32)
            if vec.shape[0] < self._total_classes:
                vec = np.pad(vec, (0, self._total_classes - vec.shape[0]), mode="constant")
            elif vec.shape[0] > self._total_classes:
                vec = vec[: self._total_classes]
            self.combo_id_to_multihot[combo_id] = vec

        self.sample_idx_to_combo = {}
        for idx, label_vec in enumerate(labels):
            key = self._multihot_to_key(label_vec)
            self.sample_idx_to_combo[idx] = self.combo_to_id[key]

    def _align_labels(self, labels: torch.Tensor, out_dim: int) -> torch.Tensor:
        if labels.dim() == 1:
            labels = F.one_hot(labels.long(), num_classes=out_dim).float()
        elif labels.dim() == 2 and labels.size(1) == 1:
            labels = F.one_hot(labels.squeeze(1).long(), num_classes=out_dim).float()
        else:
            labels = labels.float()

        if labels.size(1) < out_dim:
            labels = F.pad(labels, (0, out_dim - labels.size(1)), mode="constant", value=0.0)
        elif labels.size(1) > out_dim:
            labels = labels[:, :out_dim]
        return labels

    @staticmethod
    def _split_budget(total_budget: int, num_parts: int) -> List[int]:
        total_budget = max(int(total_budget), 0)
        num_parts = max(int(num_parts), 0)
        if num_parts == 0:
            return []

        base, remainder = divmod(total_budget, num_parts)
        return [base + 1 if idx < remainder else base for idx in range(num_parts)]

    def _task_syn_dir(self, task_id: int) -> str:
        return os.path.join(self.wsdm_save_path, f"task_{task_id}")

    def _task_model_file(self, client_id: int) -> str:
        return os.path.join(
            self.wsdm_save_path + "_model",
            f"task_{self._cur_task}",
            f"_client{client_id}.pt",
        )

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_dataset = self.dataset_names[self._cur_task]
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        info = "Learning on {}: {}-{}".format(self._cur_dataset, self._known_classes, self._total_classes)
        print(info)
        self.logger.log(info)

        train_dataset = data_manager.get_dataset(begin=self._cur_task, end=self._cur_task, mode="train")
        val_datasets = data_manager.get_dataset(begin=0, end=self._cur_task, mode="val")
        test_datasets = data_manager.get_dataset(begin=0, end=self._cur_task, mode="test")

        self.val_loader = [self._build_eval_loader(ds) for ds in val_datasets]
        self.test_loader = [self._build_eval_loader(ds) for ds in test_datasets]
        if len(self.val_loader) == 0:
            self.val_loader = self.test_loader

        train_set = train_dataset[0]
        train_labels = self._dataset_labels_as_multihot(train_set)
        setup_seed(self.seed)
        user_groups, _ = partition_data(
            train_labels,
            beta=self.args["beta"],
            n_parties=self.args["num_users"],
        )

        self.local_train_loaders = []
        loader_kwargs = self._loader_kwargs(shuffle=True)
        for idx in range(self.args["num_users"]):
            split_dataset = IndexedDatasetSplit(train_set, user_groups[idx])
            self.local_train_loaders.append(DataLoader(split_dataset, **loader_kwargs))

        self._build_combo_mapping(train_set)
        self.fl_syn_train_loader = None
        self.cl_syn_train_loader = None

        self._fl_train()

        if self._cur_task != self.tasks:
            torch.save(
                deepcopy(self._network.state_dict()),
                "store/{}_{}_{}_{}_{}_{}_{}_{}.pkl".format(
                    self.args["exp_name"],
                    self._cur_dataset,
                    self.args["net"],
                    self.args["num_users"],
                    self.args["beta"],
                    self.args["method"],
                    self._cur_task,
                    self.nums,
                ),
            )

    def _local_update(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.get("local_lr", 0.01),
            momentum=0.9,
            weight_decay=self.args.get("weight_decay", 5e-4),
        )

        for _ in range(self.args["local_ep"]):
            if self.cl_syn_train_loader is not None:
                for images, labels in self.cl_syn_train_loader:
                    images = images.cuda()
                    labels = labels.cuda()
                    output = model(images)["logits"]
                    labels = self._align_labels(labels, output.shape[1])
                    loss = F.binary_cross_entropy_with_logits(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            for _, images, labels in train_data_loader:
                images = images.cuda()
                labels = labels.cuda()
                output = model(images)["logits"]
                labels = self._align_labels(labels, output.shape[1])
                loss = F.binary_cross_entropy_with_logits(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model, model.state_dict()

    def _train_syn(self, model, train_data_loader):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.args.get("local_lr", 0.01),
            momentum=0.9,
            weight_decay=self.args.get("weight_decay", 5e-4),
        )

        for images, labels in train_data_loader:
            images = images.cuda()
            labels = labels.cuda()
            output = model(images)["logits"]
            labels = self._align_labels(labels, output.shape[1])
            loss = F.binary_cross_entropy_with_logits(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model.state_dict()

    def init_syn_dataloader(self, loaders: List[DataLoader], mode: str):
        for data_loader in loaders:
            if not hasattr(data_loader.dataset, "dataset"):
                continue
            base_dataset = data_loader.dataset.dataset
            all_images = getattr(base_dataset, "image_paths", None)
            if all_images is None:
                continue

            for rr, _, _ in data_loader:
                idxs = rr.cpu().numpy().tolist()
                for idx in idxs:
                    idx = int(idx)
                    combo_id = self.sample_idx_to_combo.get(idx, -1)
                    savepath = os.path.join(
                        self.wsdm_save_path + "_" + mode,
                        "task_" + str(self._cur_task),
                        "combo_" + str(combo_id),
                    )
                    os.makedirs(savepath, exist_ok=True)
                    image_path = all_images[idx]
                    if isinstance(image_path, str) and os.path.isfile(image_path):
                        dest_path = os.path.join(savepath, os.path.basename(image_path))
                        if not os.path.exists(dest_path):
                            shutil.copy(image_path, savepath)

    def diversity_cluster(self):
        task_dir = self._task_syn_dir(self._cur_task)

        # Check whether all clients have finished generation.
        complete_clients = 0
        if os.path.isdir(task_dir):
            for li in range(self.args["num_users"]):
                meta_file = os.path.join(task_dir, f"metadata_client{li}.npz")
                if os.path.isfile(meta_file):
                    complete_clients += 1

        if complete_clients == self.args["num_users"]:
            print(f"Task {self._cur_task}: synthetic data exists for all clients ({complete_clients}); skipping clustering.")
            return

        os.makedirs(os.path.join(task_dir, "images"), exist_ok=True)

        total_budget = max(int(self.nums), 0)

        if total_budget <= 0:
            self.logger.log("WSDM synthetic replay budget is 0; skip generation.")
            return

        self._network.eval()
        client_records = []
        for li in range(len(self.local_train_loaders)):
            meta_file = os.path.join(task_dir, f"metadata_client{li}.npz")
            if os.path.isfile(meta_file):
                print(f"Client {li}: metadata exists; skipping feature extraction.")
                data = np.load(meta_file, allow_pickle=True)
                labels = data["labels"]
                client_records.append({
                    "client_id": li,
                    "unique_combo_ids": [],  # Unused when generation is skipped.
                    "num_labels": self.combo_count,
                    "savefile": self._task_model_file(li),
                })
                continue

            features = []
            idxs_list = []

            for rr, images, _ in self.local_train_loaders[li]:
                rr_np = rr.cpu().numpy()
                idxs_list.append(rr_np)
                t_images = images.cuda()
                feature = self._network.extract_vector(t_images)
                features.append(feature.cpu().detach().numpy())

            if len(features) == 0:
                continue

            features = np.concatenate(features, axis=0)
            idxs_list = np.concatenate(idxs_list, axis=0)
            combo_ids = np.array([self.sample_idx_to_combo[int(idx)] for idx in idxs_list], dtype=np.int64)
            unique_combo_ids = np.unique(combo_ids)
            num_labels = max(self.combo_count, 1)

            d_shape_label: Dict[int, int] = {}
            for combo_id in unique_combo_ids:
                idx = np.where(combo_ids == combo_id)[0]
                t_features = features[idx]
                t_idxs = idxs_list[idx]

                if len(idx) < self.cluster_num:
                    cluster_labels = np.arange(len(idx))
                else:
                    pca = PCA(n_components=min(t_features.shape[0], t_features.shape[1], 50))
                    reduced = pca.fit_transform(t_features).T
                    _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                        reduced,
                        self.cluster_num,
                        self.fuzz_m,
                        error=0.0005,
                        maxiter=10000,
                        init=None,
                    )
                    cluster_labels = np.argmax(u, axis=0)

                for i in range(self.cluster_num):
                    for j in range(len(t_idxs)):
                        if cluster_labels[j] == i:
                            d_shape_label[int(t_idxs[j])] = int(i + num_labels + 2)

            if len(d_shape_label) == 0:
                continue

            savefile = self.train_wsdm(d_shape_label, num_labels, self.local_train_loaders[li], li)
            client_records.append(
                {
                    "client_id": li,
                    "unique_combo_ids": unique_combo_ids,
                    "num_labels": num_labels,
                    "savefile": savefile,
                }
            )

        if len(client_records) == 0:
            self.logger.log("WSDM found no eligible client for synthetic replay generation.")
            return

        client_budgets = self._split_budget(total_budget, len(client_records))
        generated_total = 0
        for record, client_budget in zip(client_records, client_budgets):
            client_budget = int(client_budget)
            if client_budget <= 0:
                continue

            combo_budgets = self._split_budget(client_budget, len(record["unique_combo_ids"]))
            combo_budget_map = {
                int(combo_id): int(combo_budget)
                for combo_id, combo_budget in zip(record["unique_combo_ids"], combo_budgets)
            }
            generated_total += self.replay_syn_data(
                record["unique_combo_ids"],
                record["num_labels"],
                record["savefile"],
                record["client_id"],
                combo_budget_map,
            )

        if generated_total != total_budget:
            raise RuntimeError(
                "WSDM replay count mismatch: expected {}, got {} on task {}".format(
                    total_budget,
                    generated_total,
                    self._cur_task,
                )
            )

    def train_wsdm(self, d_shape_label: Dict[int, int], num_labels: int, train_loader: DataLoader, client_id: int):
        savefile = self._task_model_file(client_id)
        if os.path.isfile(savefile):
            print(f"Task {self._cur_task} client {client_id}: diffusion model exists; skipping training.")
            return savefile

        num_samples = max(int(getattr(train_loader.sampler, "num_samples", len(train_loader.dataset))), 1)

        total_rounds = wsdm_com_rounds

        local_ep = wsdm_local_ep
        epochs = total_rounds * local_ep

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Diffusion training is much more memory intensive than local classifier training.
        # Use a dedicated small batch loader to avoid OOM on high-resolution settings.
        diffusion_batch_size = max(1, self.wsdm_train_batch_size)
        diffusion_dataset = DiffusionIndexedDataset(
            train_loader.dataset.dataset,
            train_loader.dataset.idxs,
            self.diffusion_transform,
        )
        diffusion_loader = DataLoader(
            diffusion_dataset,
            batch_size=diffusion_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        net_model = UNet(
            T=self.diffusion_steps,
            num_labels=num_labels,
            num_shapes=self.cluster_num + num_labels + 1,
            pca_fcel=False,
            embedding_type=1,
            ch=128,
            ch_mult=[1, 2, 3, 4],
            aen=True,
            attn=[2],
            num_res_blocks=2,
            dropout=0.15,
        ).to(device)

        optimizer = torch.optim.AdamW(net_model.parameters(), lr=1e-4, weight_decay=1e-4)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs,
            eta_min=0,
            last_epoch=-1,
        )
        warmup_scheduler = GradualWarmupScheduler(
            optimizer=optimizer,
            multiplier=2,
            warm_epoch=max(1, epochs // 10),
            after_scheduler=cosine_scheduler,
        )
        trainer = GaussianDiffusionTrainer(net_model, 1e-4, 0.02, self.diffusion_steps).to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.wsdm_amp and device.type == "cuda"))

        prog_bar = tqdm(range(epochs))
        for _, com in enumerate(prog_bar):
            for rr, images, _ in diffusion_loader:
                optimizer.zero_grad()

                x_0 = images.to(device)

                idxs = rr.cpu().numpy().tolist()
                cls_label = [self.sample_idx_to_combo[int(idx)] for idx in idxs]
                shape_label = [d_shape_label.get(int(idx), num_labels + 1) for idx in idxs]

                cls_label = torch.LongTensor(cls_label).to(device) + 1
                shape_label = torch.LongTensor(shape_label).to(device)

                if np.random.rand() < 0.1:
                    cls_label = torch.zeros_like(cls_label).to(device)
                    shape_label = torch.zeros_like(shape_label).to(device) + num_labels + 1

                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        loss = trainer(x_0, cls_label, shape_label).mean()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net_model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = trainer(x_0, cls_label, shape_label).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_model.parameters(), 1)
                    optimizer.step()

                del x_0, cls_label, shape_label, loss

            warmup_scheduler.step()
            info = "Task {}, Epoch {}/{}".format(self._cur_task, com + 1, epochs)
            prog_bar.set_description(info)

        savefile = self._task_model_file(client_id)
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        torch.save(net_model.state_dict(), savefile)
        return savefile

    def replay_syn_data(
            self,
            unique_labels: np.ndarray,
            num_labels: int,
            ckptfile: str,
            client_id: int,
            combo_budget_map: Dict[int, int],
    ):
        task_dir = self._task_syn_dir(self._cur_task)
        meta_file = os.path.join(task_dir, "metadata_client{}.npz".format(client_id))

        if os.path.isfile(meta_file):
            print(f"Task {self._cur_task} client {client_id}: metadata exists; skipping image generation.")
            data = np.load(meta_file, allow_pickle=True)
            return len(data["paths"])

        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = UNet(
                T=self.diffusion_steps,
                num_labels=num_labels,
                num_shapes=self.cluster_num + num_labels + 1,
                pca_fcel=False,
                embedding_type=1,
                ch=128,
                ch_mult=[1, 2, 3, 4],
                aen=True,
                attn=[2],
                num_res_blocks=2,
                dropout=0.15,
            ).to(device)
            ckpt = torch.load(ckptfile, map_location=device)
            model.load_state_dict(ckpt)
            model.eval()

            sampler = GaussianDiffusionSampler(model, 1e-4, 0.02, self.diffusion_steps, w=0.3,
                                               num_labels=num_labels).to(device)

            image_dir = os.path.join(task_dir, "images")
            os.makedirs(image_dir, exist_ok=True)

            saved_paths: List[str] = []
            saved_labels: List[np.ndarray] = []
            generated_count = 0
            cluster_count = max(int(self.cluster_num), 1)
            batch_size = max(int(self.g_batch_size), 1)

            prog_bar = tqdm(unique_labels)
            for _, cls_label in enumerate(prog_bar):
                combo_budget = int(combo_budget_map.get(int(cls_label), 0))
                remain = max(combo_budget, 0)
                shape_offset = 0

                while remain > 0:
                    current_batch = min(batch_size, remain)
                    shape_label = shape_offset % cluster_count

                    cls_labels = torch.full((current_batch,), int(cls_label) + 1, dtype=torch.long, device=device)
                    shape_labels = torch.full(
                        (current_batch,),
                        int(shape_label) + 2 + num_labels,
                        dtype=torch.long,
                        device=device,
                    )

                    noisy_image = torch.randn(
                        size=[current_batch, 3, self.diffusion_image_size, self.diffusion_image_size],
                        device=device,
                    )
                    sampled_imgs = sampler(noisy_image, cls_labels, shape_labels)
                    sampled_imgs = sampled_imgs * 0.5 + 0.5

                    for i in range(len(sampled_imgs)):
                        ndarr = (
                            sampled_imgs[i]
                            .mul(255)
                            .add_(0.5)
                            .clamp_(0, 255)
                            .permute(1, 2, 0)
                            .to("cpu", torch.uint8)
                            .numpy()
                        )
                        im = Image.fromarray(ndarr)
                        filename = (
                            "client{}_combo{}_diverse{}_{}.jpg".format(
                                client_id,
                                int(cls_label),
                                int(shape_label),
                                generated_count + i,
                            )
                        )
                        abs_path = os.path.join(image_dir, filename)
                        im.save(abs_path, format=None)

                        rel_path = os.path.relpath(abs_path, task_dir)
                        saved_paths.append(rel_path)
                        label_vec = self.combo_id_to_multihot.get(
                            int(cls_label),
                            np.zeros(self._total_classes, dtype=np.float32),
                        )
                        saved_labels.append(label_vec.copy())

                    generated_count += len(sampled_imgs)
                    remain -= len(sampled_imgs)
                    shape_offset += 1

                info = "Task {}, replay_combo {}/{}".format(self._cur_task, int(cls_label), num_labels)
                prog_bar.set_description(info)

            if len(saved_paths) > 0:
                meta_file = os.path.join(task_dir, "metadata_client{}.npz".format(client_id))
                np.savez(
                    meta_file,
                    paths=np.array(saved_paths),
                    labels=np.stack(saved_labels).astype(np.float32),
                )

            return generated_count

    def init_cl_syn_loader(self):
        self.cl_syn_train_loader = None
        if self._cur_task <= 0:
            return

        datasets = []
        for t in range(self._cur_task):
            syn_dataset = WSDMSynDataset(
                self._task_syn_dir(t),
                transform=self.syn_transform,
                target_dim=self._total_classes,
            )
            if len(syn_dataset) > 0:
                datasets.append(syn_dataset)

        if len(datasets) == 0:
            return

        merged = ConcatDataset(datasets)
        kwargs = self._loader_kwargs(shuffle=True)
        self.cl_syn_train_loader = DataLoader(merged, **kwargs)

    def init_fl_syn_loader(self):
        self.fl_syn_train_loader = None

        datasets = []
        for t in range(self._cur_task + 1):
            syn_dataset = WSDMSynDataset(
                self._task_syn_dir(t),
                transform=self.syn_transform,
                target_dim=self._total_classes,
            )
            if len(syn_dataset) > 0:
                datasets.append(syn_dataset)

        if len(datasets) == 0:
            return

        merged = ConcatDataset(datasets)
        kwargs = self._loader_kwargs(shuffle=True)
        self.fl_syn_train_loader = DataLoader(merged, **kwargs)

    def _fl_train(self):
        if self._cur_task > 0:
            self.init_cl_syn_loader()

        self._network.cuda()
        best_model = deepcopy(self._network.state_dict())
        best_val_auc = float("-inf")

        total_rounds = self.args.get("com_rounds", None)
        if isinstance(total_rounds, list):
            total_rounds = total_rounds[self._cur_task]
        elif total_rounds is None:
            total_rounds = self.args.get("com_round", 100)

        mid_round = min(total_rounds - 1, max(0, int(total_rounds * self.mid_round_ratio)))

        start_round = 0
        checkpoint = self._load_checkpoint(optimizer=None, scheduler=None)
        if checkpoint is not None:
            start_round = int(checkpoint.get("com_round", 0))
            print(f"Resuming from checkpoint: task {self._cur_task}, round {start_round}/{total_rounds}")
            if getattr(self, 'logger', None) is not None:
                self.logger.log(f"Resume checkpoint: task {self._cur_task}, com_round {start_round}/{total_rounds}")
            best_model = checkpoint.get("best_model", best_model)
            best_val_auc = checkpoint.get("best_val_auc", best_val_auc)

        if start_round >= total_rounds:
            self._network.load_state_dict(best_model)
            torch.cuda.empty_cache()
            return

        # Recover generation when resuming after mid_round without generated images.
        if start_round > mid_round:
            if self.fl_syn_train_loader is None:
                print(
                    f"Task {self._cur_task}: resuming at round {start_round} after mid_round {mid_round}; recovering generation...")
                self.diversity_cluster()
                self.init_syn_dataloader(self.local_train_loaders, "train")
                self.init_fl_syn_loader()

        prog_bar = tqdm(range(start_round, total_rounds))
        for _, com in enumerate(prog_bar):
            # Runtime check to prevent accidental skipping.
            if com > mid_round and self.fl_syn_train_loader is None:
                print(f"com={com} is past mid_round but synthetic data is not loaded; recovering...")
                self.diversity_cluster()
                self.init_syn_dataloader(self.local_train_loaders, "train")
                self.init_fl_syn_loader()

            local_weights = []

            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)

            for idx in idxs_users:
                train_net = deepcopy(self._network)
                train_net, w = self._local_update(train_net, self.local_train_loaders[idx])
                if self.fl_syn_train_loader is not None:
                    w = self._train_syn(train_net, self.fl_syn_train_loader)
                local_weights.append(deepcopy(w))
                del train_net

            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            val_auc = self._compute_auc(self._network, self.val_loader)
            if val_auc >= best_val_auc:
                best_val_auc = val_auc
                best_model = deepcopy(self._network.state_dict())
                self.best_model = best_model
                self.best_val_auc = best_val_auc

            self._save_checkpoint(com + 1, optimizer=None, scheduler=None)

            info = "Task {}, Epoch {}/{} => Val_AUC {:.4f}".format(
                self._cur_task,
                com + 1,
                total_rounds,
                val_auc,
            )
            prog_bar.set_description(info)

            if com == mid_round:
                self.diversity_cluster()
                self.init_syn_dataloader(self.local_train_loaders, "train")
                self.init_fl_syn_loader()

        self._network.load_state_dict(best_model)
        torch.cuda.empty_cache()
