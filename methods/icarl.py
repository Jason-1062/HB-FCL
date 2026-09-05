import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from methods.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb
from PIL import Image

EPSILON = 1e-8
T = 2


class CombinedDataset(torch.utils.data.Dataset):
    """
    Lazy combination of current-task data and random subsets from previous tasks.
    Images are loaded on demand, and label padding is performed in __getitem__.
    """

    def __init__(self, pre_dataset, cur_dataset, size, cur_label_dim):
        self.pre_dataset = pre_dataset
        self.cur_dataset = cur_dataset
        self.cur_label_dim = cur_label_dim
        self.cur_len = len(cur_dataset)
        self.pre_len = len(pre_dataset)

        if size < self.pre_len:
            self.pre_idxs = torch.randperm(self.pre_len)[:size].tolist()
            self.use_all_pre = False
        else:
            self.pre_idxs = list(range(self.pre_len))
            self.use_all_pre = True

    def __len__(self):
        return self.cur_len + len(self.pre_idxs)

    def __getitem__(self, idx):
        if idx < self.cur_len:
            image, label = self.cur_dataset[idx]
        else:
            pre_idx = self.pre_idxs[idx - self.cur_len]
            image, label = self.pre_dataset[pre_idx]
            # Pad previous-task labels to the current label dimension.
            if label.shape[0] < self.cur_label_dim:
                pad_size = self.cur_label_dim - label.shape[0]
                label = F.pad(label, (0, pad_size), mode="constant", value=0.0)
        return image, label


def print_data_stats(client_id, train_data_loader):
    """Print compact dataset statistics for multi-label data."""
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(train_data_loader):
        total_samples += images.shape[0]
    return f"total_samples: {total_samples}"


class iCaRL(BaseLearner):
    def __init__(self, args, logger=None):
        super().__init__(args, logger)
        self._network = IncrementalNet(args, False, False)
        self.memory_size = args["nums"]

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def combine_dataset(self, pre_dataset, cur_dataset, size):
        """
        Combine previous and current datasets with lazy loading to avoid OOM.

        Images are not stacked in memory; CombinedDataset reads and pads labels on demand.
        """
        # Get the current label dimension for padding old labels.
        sample_img, sample_label = cur_dataset[0]
        cur_label_dim = sample_label.shape[0]

        combined = CombinedDataset(pre_dataset, cur_dataset, size, cur_label_dim)
        return combined

    def get_all_previous_dataset(self, data_manager, idx):
        """
        Get training data from previous tasks for this client using lazy loading.

        Return lazy splits instead of stacking all previous data in memory.
        Each DatasetSplit stores indices only; images are read in __getitem__.
        """
        all_previous_splits = []

        for task_idx in range(self._cur_task):
            task_datasets = data_manager.get_dataset(
                begin=task_idx, end=task_idx, mode="train"
            )

            for dataset in task_datasets:
                labels = dataset.muti_labels
                setup_seed(self.seed)
                user_groups, _ = partition_data(
                    labels, beta=self.args["beta"], n_parties=self.args["num_users"]
                )
                client_dataset = DatasetSplit(dataset, user_groups[idx])
                all_previous_splits.append(client_dataset)

        if len(all_previous_splits) == 0:
            return None

        if len(all_previous_splits) == 1:
            return all_previous_splits[0]

        # Lazy combination: access DatasetSplit only in __getitem__.
        return ConcatDataset(all_previous_splits)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_dataset = self.dataset_names[self._cur_task]
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Get current-task training datasets.
        train_datasets = data_manager.get_dataset(
            begin=self._cur_task, end=self._cur_task, mode="train"
        )

        # Get validation datasets.
        val_datasets = data_manager.get_dataset(begin=0, end=self._cur_task, mode="val")

        # Get previous-task test datasets for evaluation.
        test_datasets = data_manager.get_dataset(
            begin=0, end=self._cur_task, mode="test"
        )

        # Create validation data loaders.
        self.val_loader = []
        for val_dataset in val_datasets:
            val_loader = DataLoader(
                val_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=self.args["num_worker"],
                multiprocessing_context=self.args["mulc"],
                persistent_workers=True,
            )
            self.val_loader.append(val_loader)

        # Create test data loaders.
        self.test_loader = []
        for test_dataset in test_datasets:
            test_loader = DataLoader(
                test_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=self.args["num_worker"],
                multiprocessing_context=self.args["mulc"],
                persistent_workers=True,
            )
            self.test_loader.append(test_loader)

        self._network.cuda()
        setup_seed(self.seed)
        self._fl_train(train_datasets, self.val_loader, self.test_loader, data_manager)

        # Save the model.
        if self._cur_task != self.tasks:
            torch.save(
                copy.deepcopy(self._network.state_dict()),
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

    def _local_update(self, model, train_data_loader, lr):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )
        for iter in range(self.args["local_ep"]):
            for batch_idx, (images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                # Multi-label classification uses binary_cross_entropy_with_logits.
                # Match target and output dimensions.
                if labels.dim() == 1:
                    # Convert single labels to one-hot multi-label form.
                    labels = F.one_hot(labels, num_classes=output.shape[1]).float()
                elif labels.shape[1] < output.shape[1]:
                    # Pad shorter targets with zeros.
                    pad_cols = output.shape[1] - labels.shape[1]
                    labels = F.pad(labels, (0, pad_cols), mode="constant", value=0.0)
                elif labels.shape[1] > output.shape[1]:
                    # Truncate longer targets.
                    labels = labels[:, : output.shape[1]]

                loss = F.binary_cross_entropy_with_logits(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def _local_finetune(self, model, train_data_loader, lr):
        model.train()
        self._old_network.eval()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )

        for iter in range(self.args["local_ep"]):
            for batch_idx, (images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]

                # Match target and output dimensions.
                if labels.dim() == 1:
                    # Convert single labels to one-hot multi-label form.
                    labels = F.one_hot(labels, num_classes=output.shape[1]).float()
                elif labels.shape[1] < output.shape[1]:
                    # Pad shorter targets with zeros.
                    pad_cols = output.shape[1] - labels.shape[1]
                    labels = F.pad(labels, (0, pad_cols), mode="constant", value=0.0)
                elif labels.shape[1] > output.shape[1]:
                    # Truncate longer targets.
                    labels = labels[:, : output.shape[1]]

                # Fine-tune on new tasks with binary_cross_entropy_with_logits.
                loss_clf = F.binary_cross_entropy_with_logits(output, labels)

                # Distill only classes known before the current task.
                with torch.no_grad():
                    old_output = self._old_network(images)["logits"]

                # Match old_output to the first known_classes outputs.
                if old_output.shape[1] > self._known_classes:
                    old_output = old_output[:, : self._known_classes]
                elif old_output.shape[1] < self._known_classes:
                    pad_cols = self._known_classes - old_output.shape[1]
                    old_output = F.pad(
                        old_output, (0, pad_cols), mode="constant", value=0.0
                    )

                loss_kd = _KD_loss(
                    output[:, : self._known_classes],
                    old_output,
                    T,
                )

                loss = loss_clf + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def _fl_train(self, train_datasets, val_loader, test_loader, data_manager):
        """
        Federated training for multiple multi-label datasets.
        Select the model with the best validation macro F1.

        Args:
            train_datasets: Training datasets for the current task.
            val_loader: Validation data loaders.
            test_loader: Test data loaders.
            data_manager: Dataset manager.
        """
        self._network.cuda()
        best_model = copy.deepcopy(self._network.state_dict())
        best_val_auc = float("-inf")

        # Combine all current-task training datasets.
        if len(train_datasets) == 0:
            raise ValueError("No training datasets for current task")
        elif len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            train_dataset = ConcatDataset(train_datasets)

        labels = train_dataset.muti_labels

        user_groups, _ = partition_data(
            labels, beta=self.args["beta"], n_parties=self.args["num_users"]
        )

        # Get communication rounds.
        total_rounds = self.args.get("com_rounds", None)
        total_rounds = total_rounds[self._cur_task]

        optimizer = torch.optim.SGD(
            self._network.parameters(),
            lr=self.args["local_lr"],
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 75], gamma=0.1
        )

        start_round = 0
        checkpoint = self._load_checkpoint(optimizer, scheduler)
        if checkpoint is not None:
            start_round = int(checkpoint.get("com_round", 0))
            print(
                f"Resuming from checkpoint: task {self._cur_task}, round {start_round}/{total_rounds}"
            )
            if self.logger is not None:
                self.logger.log(
                    f"Resume checkpoint: task {self._cur_task}, com_round {start_round}/{total_rounds}"
                )
            best_model = checkpoint.get("best_model", best_model)
            best_val_auc = checkpoint.get("best_val_auc", best_val_auc)

        if start_round >= total_rounds:
            self._network.load_state_dict(best_model)
            torch.cuda.empty_cache()
            return

        prog_bar = tqdm(range(start_round, total_rounds))

        for com in prog_bar:
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(
                range(self.args["num_users"]), m, replace=False
            )
            for idx in idxs_users:
                # update local train data
                if self._cur_task == 0:
                    local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                else:
                    # Get the current local dataset.
                    current_local_dataset = DatasetSplit(
                        train_dataset, user_groups[idx]
                    )

                    # Get previous-task data for this client.
                    previous_local_dataset = self.get_all_previous_dataset(
                        data_manager, idx
                    )

                    if previous_local_dataset is not None:
                        # Combine previous and current datasets lazily.
                        local_dataset = self.combine_dataset(
                            previous_local_dataset,
                            current_local_dataset,
                            self.memory_size // self.args["num_users"],
                        )
                    else:
                        # Use only current-task data when no previous data exists.
                        local_dataset = current_local_dataset

                local_train_loader = DataLoader(
                    local_dataset,
                    batch_size=self.args["local_bs"],
                    shuffle=True,
                    num_workers=self.args["num_worker"],
                    pin_memory=True,
                    multiprocessing_context=self.args["mulc"],
                    persistent_workers=True,
                )

                current_lr = scheduler.get_last_lr()[0]
                if self._cur_task == 0:
                    w = self._local_update(
                        copy.deepcopy(self._network), local_train_loader, lr=current_lr
                    )
                else:
                    w = self._local_finetune(
                        copy.deepcopy(self._network), local_train_loader, lr=current_lr
                    )
                local_weights.append(copy.deepcopy(w))
                del local_train_loader, local_dataset, w
                # torch.cuda.empty_cache()

            scheduler.step()

            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            # Evaluate on validation data and select the best model.
            val_auc = self._compute_auc(copy.deepcopy(self._network), val_loader)
            print(
                "Task {}, Epoch {}/{} =>  Val_F1 {:.4f}".format(
                    self._cur_task, com + 1, total_rounds, val_auc
                )
            )
            if val_auc >= best_val_auc:
                best_val_auc = val_auc
                best_model = copy.deepcopy(self._network.state_dict())
                self.best_model = best_model
                self.best_val_auc = best_val_auc

            self._save_checkpoint(com + 1, optimizer, scheduler)

        self._network.load_state_dict(best_model)
        del best_model
        torch.cuda.empty_cache()


def _KD_loss(pred, soft, T):
    return F.binary_cross_entropy_with_logits(pred / T, torch.sigmoid(soft / T)) * (
        T**2
    )
