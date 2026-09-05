"""
FedRNC: Federated Robust Noisy-label Correction (Multi-label Adapted)

Three-phase pipeline per task:
  Phase 1 - GMM Noise Detection: per-sample BCE loss → GMM → exclude noise-suspects
  Phase 2 - FL Training: FedAvg on cleaned data
  Phase 3 - Prototype-Guided Relabeling: feature extraction → global prototypes →
            cross-label analysis → selective label-bit flipping
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm import tqdm

from methods.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, setup_seed

EPSILON = 1e-8


class FedRNCDataStore:
    """Manage a single client's data, supporting noise detection and relabeling."""

    def __init__(self, data_list):
        """
        Args:
            data_list: list of (image_tensor, label_tensor) tuples
        """
        self.data = list(data_list)
        self.excluded_data = []
        self.noise_indices = set()

    def exclude_noisy(self, noise_indices):
        """Move noisy samples from data to excluded_data."""
        self.noise_indices = set(noise_indices)
        new_data = []
        for i, item in enumerate(self.data):
            if i in self.noise_indices:
                self.excluded_data.append(item)
            else:
                new_data.append(item)
        self.data = new_data

    def relabel_samples(self, relabel_map, matched_label, task_label_indices):
        """
        Selective label-bit flipping on excluded_data (without merging).

        Args:
            relabel_map: set of excluded_data indices that should be relabeled
            matched_label: the label index that should be flipped from 0 to 1
            task_label_indices: list of label indices in the current task group
        """
        for idx in range(len(self.excluded_data)):
            if idx not in relabel_map:
                continue
            image, label = self.excluded_data[idx]
            if isinstance(label, torch.Tensor):
                label = label.clone()
            else:
                label = torch.tensor(label, dtype=torch.float32)

            # Add the matched (true) label: 0 -> 1
            label[matched_label] = 1.0
            # Remove suspected noisy labels within task group: 1 -> 0
            for tl in task_label_indices:
                if tl != matched_label and label[tl] > 0.5:
                    label[tl] = 0.0
            self.excluded_data[idx] = (image, label)

    def merge_excluded(self):
        """Merge all excluded_data back into data after relabeling is done."""
        self.data.extend(self.excluded_data)
        self.excluded_data = []

    def get_loader(self, batch_size, num_workers, mulc, shuffle=True):
        return DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mulc,
            persistent_workers=False,
        )


class FedRNC(BaseLearner):
    """FedRNC: Federated Robust Noisy-label Correction for Multi-label CL."""

    def __init__(self, args, logger):
        super().__init__(args, logger)
        self._network = IncrementalNet(args, False)
        self._old_network = None

        # FedRNC-specific hyperparameters
        self.when_stage2added = args.get("when_stage2added", 2)
        self.warmup_end = args.get("warmup_end", 5)
        self.pretrain_rounds = args.get("pretrain_rounds", 5)

        # Per-client data stores
        self._client_stores = {}

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_dataset = self.dataset_names[self._cur_task]
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        print(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # Get datasets
        train_datasets = data_manager.get_dataset(
            begin=self._cur_task, end=self._cur_task, mode="train"
        )
        val_datasets = data_manager.get_dataset(
            begin=0, end=self._cur_task, mode="val"
        )
        test_datasets = data_manager.get_dataset(
            begin=0, end=self._cur_task, mode="test"
        )

        self.val_loader = []
        for val_dataset in val_datasets:
            self.val_loader.append(
                DataLoader(
                    val_dataset,
                    batch_size=256,
                    shuffle=False,
                    num_workers=self.args["num_worker"],
                    multiprocessing_context=self.args["mulc"],
                    persistent_workers=False,
                )
            )

        self.test_loader = []
        for test_dataset in test_datasets:
            self.test_loader.append(
                DataLoader(
                    test_dataset,
                    batch_size=256,
                    shuffle=False,
                    num_workers=self.args["num_worker"],
                    multiprocessing_context=self.args["mulc"],
                    persistent_workers=False,
                )
            )

        self._network.cuda()
        setup_seed(self.seed)
        self._fl_train(train_datasets, self.val_loader, self.test_loader, data_manager)

        # Save model
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

    # =========================================================================
    # Phase 1: GMM Noise Detection
    # =========================================================================

    def _local_pretrain_with_noise_detection(
            self, model, train_loader, lr, return_weights=False
    ):
        """
        Phase 1: Local training with per-sample loss tracking and GMM detection.
        Returns:
            - model state_dict if return_weights=True
            - set of noise-suspect sample indices otherwise
        """
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args["weight_decay"]
        )

        all_sample_losses = {}  # global_idx -> list of per-sample losses across iterations
        sample_offset = 0

        for iteration in range(self.args["local_ep"]):
            batch_offset = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.cuda(), labels.cuda()
                labels = self._align_labels(labels, model(images)["logits"].shape[1])

                # Forward-backward-update
                output = model(images)["logits"]
                loss = F.binary_cross_entropy_with_logits(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute per-sample loss on the updated model
                with torch.no_grad():
                    output = model(images)["logits"]
                    per_sample_loss = F.binary_cross_entropy_with_logits(
                        output, labels, reduction="none"
                    ).sum(dim=1)  # [batch_size] - sum over label dimension for multi-label

                # Record losses
                for i in range(images.shape[0]):
                    global_idx = sample_offset + batch_offset + i
                    if global_idx not in all_sample_losses:
                        all_sample_losses[global_idx] = []
                    all_sample_losses[global_idx].append(
                        per_sample_loss[i].item()
                    )

                batch_offset += images.shape[0]

            sample_offset += batch_offset

        # Average per-sample losses across iterations
        avg_losses = {
            idx: np.mean(losses) for idx, losses in all_sample_losses.items()
        }

        # GMM noise detection
        noise_indices = self._detect_noisy_samples(avg_losses)
        if return_weights:
            return copy.deepcopy(model.state_dict())
        return set(noise_indices)

    def _detect_noisy_samples(self, sample_losses):
        """
        GMM-based noise detection (core algorithm unchanged from original FedRNC).

        Args:
            sample_losses: dict {sample_idx: average_loss}
        Returns:
            list of noise-suspect sample indices
        """
        if len(sample_losses) == 0:
            return []

        indices = list(sample_losses.keys())
        losses = np.array([sample_losses[idx] for idx in indices]).reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(losses)
        cluster_labels = gmm.predict(losses)
        noise_cluster = int(np.argmax(gmm.means_))  # high-loss cluster

        noise_count = np.sum(cluster_labels == noise_cluster)
        noise_ratio = noise_count / len(losses)
        noise_ratio = max(noise_ratio, 0.01)  # lower bound protection

        threshold = np.quantile(losses, (1 - noise_ratio) ** 0.5)
        noise_mask = (losses >= threshold).flatten()

        return [indices[i] for i in range(len(indices)) if noise_mask[i]]

    # =========================================================================
    # Phase 2: FL Local Training
    # =========================================================================

    def _local_update(self, model, train_loader, lr):
        """First task local training: standard BCE loss."""
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )
        for _ in range(self.args["local_ep"]):
            for images, labels in train_loader:
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                labels = self._align_labels(labels, output.shape[1])
                loss = F.binary_cross_entropy_with_logits(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def _local_finetune(self, model, train_loader, lr):
        """Subsequent task local training: BCE + KD loss."""
        model.train()
        self._old_network.eval()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )

        for _ in range(self.args["local_ep"]):
            for images, labels in train_loader:
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                labels = self._align_labels(labels, output.shape[1])

                loss_clf = F.binary_cross_entropy_with_logits(output, labels)

                # Knowledge distillation on known classes
                with torch.no_grad():
                    old_output = self._old_network(images)["logits"]
                if old_output.shape[1] > self._known_classes:
                    old_output = old_output[:, : self._known_classes]
                elif old_output.shape[1] < self._known_classes:
                    pad_cols = self._known_classes - old_output.shape[1]
                    old_output = F.pad(old_output, (0, pad_cols), value=0.0)

                loss_kd = self._kd_loss(
                    output[:, : self._known_classes], old_output
                )

                loss = loss_clf + loss_kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def _kd_loss(self, pred, soft, T=2):
        return (
                F.binary_cross_entropy_with_logits(pred / T, torch.sigmoid(soft / T))
                * (T ** 2)
        )

    def _average_state_dicts(self, state_dicts):
        """Average a sequence of state_dicts without keeping an extra list copy."""
        state_dicts = list(state_dicts)
        if len(state_dicts) == 0:
            return None

        avg_state = {k: v.clone() for k, v in state_dicts[0].items()}
        for state_dict in state_dicts[1:]:
            for key, value in state_dict.items():
                avg_state[key].add_(value)

        for key, value in avg_state.items():
            if torch.is_floating_point(value):
                value.div_(len(state_dicts))
            else:
                avg_state[key] = torch.div(
                    value, len(state_dicts), rounding_mode="floor"
                )

        return avg_state

    # =========================================================================
    # Phase 3: Feature Extraction & Prototype Construction
    # =========================================================================

    def _extract_features_by_label(self, model, data_list):
        """
        Extract features and organize by label (multi-label adaptation).

        In multi-label setting, a sample contributes to all labels where its value is 1.

        Args:
            model: the neural network
            data_list: list of (image_tensor, label_tensor) tuples
        Returns:
            features_by_label: dict[label_idx -> list of (feature_tensor, sample_global_idx)]
        """
        model.eval()
        features_by_label = defaultdict(list)

        loader = DataLoader(data_list, batch_size=64, shuffle=False)
        sample_offset = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.cuda()
                features = model.extract_vector(images)  # [B, feature_dim]

                for i in range(images.shape[0]):
                    feat = features[i].cpu()
                    label_vec = labels[i]
                    global_idx = sample_offset + i

                    # Multi-label: sample contributes to all positive labels
                    if isinstance(label_vec, torch.Tensor):
                        label_arr = label_vec
                    else:
                        label_arr = torch.tensor(label_vec, dtype=torch.float32)

                    for label_idx in range(label_arr.shape[0]):
                        if label_arr[label_idx] > 0.5:
                            features_by_label[label_idx].append(
                                (feat, global_idx)
                            )

                sample_offset += images.shape[0]

        return features_by_label

    def _compute_global_prototypes(self, client_features_list, num_labels):
        """
        Compute per-label global prototypes by averaging features across all clients.

        Args:
            client_features_list: list of dict[label_idx -> list[(feat, idx)]]
            num_labels: total number of labels
        Returns:
            prototypes: Tensor[num_labels, feature_dim] - each label's prototype
        """
        # Infer feature dimension from the first non-empty feature
        feature_dim = None
        for client_feats in client_features_list:
            for label_idx, feat_list in client_feats.items():
                if len(feat_list) > 0:
                    feature_dim = feat_list[0][0].shape[0]
                    break
            if feature_dim is not None:
                break

        if feature_dim is None:
            # Fallback: use network's feature dimension
            feature_dim = self._network.feature_dim

        prototypes = torch.zeros(num_labels, feature_dim)

        # Aggregate features from all clients
        aggregated = defaultdict(list)
        for client_feats in client_features_list:
            for label_idx, feat_list in client_feats.items():
                aggregated[label_idx].extend(feat_list)

        for label_idx in range(num_labels):
            if label_idx in aggregated and len(aggregated[label_idx]) > 0:
                feats = torch.stack(
                    [f for f, _ in aggregated[label_idx]]
                )
                prototypes[label_idx] = feats.mean(dim=0)

        return prototypes

    # =========================================================================
    # Phase 3: Cross-Label Noise Analysis (Core Adaptation)
    # =========================================================================

    def _analyze_label_noise(
            self,
            client_features,
            excluded_features,
            prototypes,
            task_label_indices,
            num_labels,
            topk_match=3,
    ):
        """
        Cross-label group noise analysis (multi-label adapted version of
        analyze_user_class_groups_anylen).

        Core logic preserved:
        1. Compute per-label similarity distribution (softmax of -distance to prototypes)
        2. Find shared comparable labels across the label group
        3. Geometric product of proportion vectors
        4. GMM-based noise ratio estimation for top-K matches

        Multi-label adaptations:
        - "class" → "label": positive samples of a label (label=1) instead of samples belonging to a class
        - A sample contributes to multiple labels' statistics
        - Cross-label analysis uses the same geometric product logic

        Args:
            client_features: dict[label_idx -> list[(feat, idx)]] for clean data
            excluded_features: dict[label_idx -> list[(feat, idx)]] for excluded data
            prototypes: Tensor[num_labels, feature_dim]
            task_label_indices: list of label indices in current task
            num_labels: total number of labels
            topk_match: number of top matching candidates
        Returns:
            list of analysis result dicts per matched label
        """
        group = task_label_indices
        group_set = set(group)

        # 1. Build per-label statistics (combining clean + excluded)
        stats = {}
        for label_idx in group:
            clean_feats = client_features.get(label_idx, [])
            excl_feats = excluded_features.get(label_idx, [])

            all_items = clean_feats + excl_feats
            if len(all_items) == 0:
                continue

            features_tensor = torch.stack([f for f, _ in all_items])
            idx_list = [idx for _, idx in all_items]

            # Compute distances to all prototypes → softmax similarity
            dists = self._compute_sqeuclidean(features_tensor, prototypes)
            scores = torch.softmax(-dists, dim=1)  # smaller distance → higher score

            # Statistics: proportion of positive samples most similar to each label
            top_labels = torch.argmax(scores, dim=1)
            counts = torch.bincount(top_labels, minlength=num_labels)
            proportions = counts.float() / max(len(all_items), 1)

            stats[label_idx] = {
                "counts": counts,
                "proportions": proportions,
                "total": len(all_items),
                "features": features_tensor,
                "idx_list": idx_list,
            }

        # 2. Find shared comparable labels (present in all group members, excluding self)
        confidence_sets = {}
        for label_idx in group:
            if label_idx not in stats:
                continue
            prop = stats[label_idx]["proportions"].clone()
            prop[label_idx] = 0  # exclude self
            nonzero = (prop > 0).nonzero(as_tuple=True)[0].tolist()
            confidence_sets[label_idx] = set(nonzero)

        if not confidence_sets:
            return []

        shared_labels = set.intersection(*confidence_sets.values())
        # Exclude labels already in the group
        shared_labels = sorted(list(shared_labels - group_set))
        if not shared_labels:
            return []

        # 3. Geometric product of proportion vectors (core logic unchanged)
        final_scores = torch.ones(len(shared_labels))
        for label_idx in group:
            if label_idx not in stats:
                continue
            p = stats[label_idx]["proportions"][shared_labels]
            p = p / (p.sum() + EPSILON)
            final_scores *= p

        # 4. Top-K matching
        sorted_scores, sorted_idx = torch.sort(final_scores, descending=True)
        top_matched = [
            shared_labels[i] for i in sorted_idx[:topk_match].tolist()
        ]
        score_dict = {
            label_idx: final_scores[i].item()
            for i, label_idx in enumerate(shared_labels)
        }

        # 5. Estimate noise ratio for each matched label (GMM on distance difference)
        results = []
        for matched_label in top_matched:
            noise_ratio = self._estimate_noise_for_matched_label(
                stats, group, matched_label, prototypes
            )
            match_score = score_dict[matched_label]
            product = noise_ratio * match_score

            results.append(
                {
                    "matched_label": matched_label,
                    "noise_ratio": noise_ratio,
                    "match_score": match_score,
                    "product": product,
                }
            )

        return results

    def _estimate_noise_for_matched_label(
            self, stats, group_labels, matched_label, prototypes
    ):
        """
        GMM-based noise ratio estimation for a matched label.

        For each sample in the label group, compute:
          distance_to_matched_prototype - min_distance_to_other_prototypes
        Then fit GMM on the differences to estimate noise ratio.

        (Multi-label adapted version of estimate_noise_for_match_class)
        """
        # Collect features of all positive samples in the group
        all_features = []
        for label_idx in group_labels:
            if label_idx in stats and stats[label_idx].get("features") is not None:
                all_features.append(stats[label_idx]["features"])

        if len(all_features) == 0:
            return 0.0

        features = torch.cat(all_features, dim=0)
        dists = self._compute_sqeuclidean(features, prototypes)

        # Distance to matched label prototype
        dist_to_match = dists[:, matched_label]

        # Minimum distance to other prototypes
        mask_other = torch.ones(prototypes.shape[0], dtype=torch.bool)
        mask_other[matched_label] = False
        other_distances = dists[:, mask_other]
        min_other_dist, _ = torch.min(other_distances, dim=1)

        # Distance difference
        dist_diffs = (dist_to_match - min_other_dist).cpu().numpy().reshape(-1, 1)

        if len(dist_diffs) < 2:
            return 0.0

        # GMM on distance differences
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(dist_diffs)
        match_component = 0 if gmm.means_[0] < gmm.means_[1] else 1
        match_probs = gmm.predict_proba(dist_diffs)[:, match_component]
        match_ratio = match_probs.mean()
        if match_ratio < 0.5:
            match_ratio = 1 - match_ratio
        return 1 - match_ratio

    # =========================================================================
    # Phase 3: Relabeling (Selective Label-Bit Flipping)
    # =========================================================================

    def _collect_noiselike_samples(
            self, client_features, target_labels, target_prototype, percentile_value
    ):
        """
        Collect sample indices whose features are closest to the target prototype.

        Args:
            client_features: dict[label_idx -> list[(feat, global_idx)]]
            target_labels: list of label indices to search within
            target_prototype: Tensor[feature_dim] - prototype of the matched label
            percentile_value: percentage of samples to select (0-100)
        Returns:
            list of sample global indices
        """
        feats_list, idx_list = [], []
        target_set = set(target_labels)

        for label_idx, items in client_features.items():
            if label_idx not in target_set or len(items) == 0:
                continue
            feats_list.extend([feat for feat, _ in items])
            idx_list.extend([idx for _, idx in items])

        if len(feats_list) == 0:
            return []

        features = torch.stack(feats_list)
        idxs = torch.tensor(idx_list, dtype=torch.long)

        # L1 distance to target prototype
        distances = torch.cdist(features, target_prototype.unsqueeze(0), p=1)
        distances = distances.squeeze(1)  # [N]

        # Select top-percentile closest samples
        n_samples = distances.numel()
        n_select = max(1, int(n_samples * percentile_value / 100.0))
        n_select = min(n_select, n_samples)
        _, top_indices = torch.topk(distances, k=n_select, largest=False, dim=0)

        return [idxs[i].item() for i in top_indices.tolist()]

    def _relabel_samples(self, data_list, sample_indices, label_idx, new_value):
        """
        Flip a specific label bit for selected samples (selective flipping).

        Args:
            data_list: list of (image_tensor, label_tensor)
            sample_indices: set of global indices to relabel
            label_idx: which label bit to flip
            new_value: the new value for that label bit (0 or 1)
        """
        for idx in sample_indices:
            if idx < 0 or idx >= len(data_list):
                continue
            image, label = data_list[idx]
            if isinstance(label, torch.Tensor):
                label = label.clone()
            else:
                label = torch.tensor(label, dtype=torch.float32)

            label[label_idx] = float(new_value)
            data_list[idx] = (image, label)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _compute_sqeuclidean(self, features, prototypes):
        """Compute squared Euclidean distance between features and prototypes."""
        return (
                (features ** 2).sum(dim=1, keepdim=True)
                - 2 * features @ prototypes.T
                + (prototypes ** 2).sum(dim=1)
        )

    def _align_labels(self, labels, output_dim):
        """Align label dimension to match output dimension."""
        if labels.dim() == 1:
            labels = F.one_hot(labels, num_classes=output_dim).float()
        elif labels.shape[1] < output_dim:
            labels = F.pad(
                labels, (0, output_dim - labels.shape[1]), value=0.0
            )
        elif labels.shape[1] > output_dim:
            labels = labels[:, :output_dim]
        return labels

    def _get_task_label_indices(self):
        """
        Get the label indices for the current task.

        In FCLCXR, each task introduces a set of labels defined by data_manager.classes.
        The global label space grows cumulatively.
        """
        task_classes = self.args["classes"]
        begin = sum(task_classes[: self._cur_task])
        end = sum(task_classes[: self._cur_task + 1])
        return list(range(begin, end))

    # =========================================================================
    # Main FL Training Loop
    # =========================================================================

    def _fl_train(self, train_datasets, val_loader, test_loader, data_manager):
        """
        Main federated training loop with FedRNC's 3-phase pipeline.

        Phase 1: Pretrain + GMM Noise Detection (when task > 0)
        Phase 2: Standard FL Training (FedAvg)
        Phase 3: Prototype-Guided Relabeling (when task > when_stage2added)
        """
        self._network.cuda()
        best_model = copy.deepcopy(self._network.state_dict())
        best_val_auc = float("-inf")

        # Merge training datasets
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

        total_rounds = self.args.get("com_rounds", None)
        total_rounds = total_rounds[self._cur_task]

        # Task label indices for Phase 3
        task_label_indices = self._get_task_label_indices()
        num_labels = self._total_classes

        # Build per-client DataStores
        self._client_stores = {}
        for client_idx in range(self.args["num_users"]):
            client_dataset = DatasetSplit(train_dataset, user_groups[client_idx])
            data_list = []
            for i in range(len(client_dataset)):
                image, label = client_dataset[i]
                data_list.append((image, label))
            self._client_stores[client_idx] = FedRNCDataStore(data_list)

        # =====================================================================
        # Phase 1: Pretrain + GMM Noise Detection (only for task > 0)
        # =====================================================================
        if self._cur_task > 0:
            print(f"\n[FedRNC] Phase 1: Noise Detection (task {self._cur_task})")
            prog_bar = tqdm(range(self.pretrain_rounds))

            for pre_round in prog_bar:
                local_weights = []
                m = max(
                    int(self.args["frac"] * self.args["num_users"]), 1
                )
                idxs_users = np.random.choice(
                    range(self.args["num_users"]), m, replace=False
                )

                for idx in idxs_users:
                    store = self._client_stores[idx]
                    loader = store.get_loader(
                        batch_size=self.args["local_bs"],
                        num_workers=self.args["num_worker"],
                        mulc=self.args["mulc"],
                        shuffle=True,
                    )
                    w = self._local_pretrain_with_noise_detection(
                        copy.deepcopy(self._network),
                        loader,
                        lr=self.args["local_lr"],
                        return_weights=True,
                    )
                    local_weights.append(w)
                    del loader

                # FedAvg
                global_weights = self._average_state_dicts(local_weights)
                self._network.load_state_dict(global_weights)
                del local_weights, global_weights

            # After Phase 1: execute noise detection results
            for idx in range(self.args["num_users"]):
                store = self._client_stores[idx]
                loader = store.get_loader(
                    batch_size=self.args["local_bs"],
                    num_workers=self.args["num_worker"],
                    mulc=self.args["mulc"],
                    shuffle=False,
                )
                noise_indices = self._local_pretrain_with_noise_detection(
                    copy.deepcopy(self._network), loader, lr=self.args["local_lr"]
                )
                if len(noise_indices) > 0:
                    store.exclude_noisy(noise_indices)
                print(
                    f"  Client {idx}: {len(noise_indices)} noisy samples detected, "
                    f"{len(store.data)} clean, {len(store.excluded_data)} excluded"
                )
                del loader

        # =====================================================================
        # Phase 2: FL Training with optional Phase 3 (Relabeling)
        # =====================================================================
        print(f"\n[FedRNC] Phase 2: FL Training (task {self._cur_task})")

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
            print(f"Resuming from checkpoint: task {self._cur_task}, round {start_round}/{total_rounds}")
            if self.logger is not None:
                self.logger.log(f"Resume checkpoint: task {self._cur_task}, com_round {start_round}/{total_rounds}")
            best_model = checkpoint.get("best_model", best_model)
            best_val_auc = checkpoint.get("best_val_auc", best_val_auc)

        if start_round >= total_rounds:
            self._network.load_state_dict(best_model)
            torch.cuda.empty_cache()
            return

        # Storage for Phase 3 analysis results (filled at warmup_end round)
        all_analysis_results = {}
        all_client_features = {}
        all_excluded_features = {}
        global_prototypes = None

        prog_bar = tqdm(range(start_round, total_rounds))
        for com in prog_bar:
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(
                range(self.args["num_users"]), m, replace=False
            )

            for idx in idxs_users:
                store = self._client_stores[idx]

                # Build training data: clean current + memory (previous tasks)
                if self._cur_task == 0:
                    train_list = store.data
                else:
                    # Get previous-task replay data for this client
                    previous_dataset = self._get_previous_data_for_client(
                        data_manager, idx
                    )
                    if previous_dataset is not None and len(previous_dataset) > 0:
                        train_list = store.data + previous_dataset
                    else:
                        train_list = store.data

                local_train_loader = DataLoader(
                    train_list,
                    batch_size=self.args["local_bs"],
                    shuffle=True,
                    num_workers=self.args["num_worker"],
                    pin_memory=True,
                    multiprocessing_context=self.args["mulc"],
                    persistent_workers=False,
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
                local_weights.append(w)
                del local_train_loader

            scheduler.step()

            # FedAvg
            global_weights = self._average_state_dicts(local_weights)
            self._network.load_state_dict(global_weights)
            del local_weights, global_weights

            # Evaluate
            val_auc = self._compute_auc(
                copy.deepcopy(self._network), val_loader
            )
            info = "Task {}, Epoch {}/{} => Val AUC {:.4f}".format(
                self._cur_task, com + 1, total_rounds, val_auc
            )
            prog_bar.set_description(info)
            if val_auc >= best_val_auc:
                best_val_auc = val_auc
                best_model = copy.deepcopy(self._network.state_dict())
                self.best_model = best_model
                self.best_val_auc = best_val_auc

            # -----------------------------------------------------------------
            # Phase 3 trigger: Feature extraction and noise analysis
            # -----------------------------------------------------------------
            if (
                    self._cur_task > self.when_stage2added
                    and com == self.warmup_end
            ):
                print(
                    f"\n[FedRNC] Phase 3: Feature Extraction & Noise Analysis"
                )

                all_client_features = {}
                all_excluded_features = {}

                for idx in idxs_users:
                    store = self._client_stores[idx]
                    # Extract features from clean data
                    all_client_features[idx] = self._extract_features_by_label(
                        self._network, store.data
                    )
                    # Extract features from excluded data
                    if len(store.excluded_data) > 0:
                        all_excluded_features[idx] = (
                            self._extract_features_by_label(
                                self._network, store.excluded_data
                            )
                        )
                    else:
                        all_excluded_features[idx] = {}

                # Compute global prototypes
                global_prototypes = self._compute_global_prototypes(
                    list(all_client_features.values()), num_labels
                )

                # Cross-label noise analysis for each client
                all_analysis_results = {}
                for idx in idxs_users:
                    print(f"\n  Client {idx} noise analysis:")
                    analysis = self._analyze_label_noise(
                        client_features=all_client_features[idx],
                        excluded_features=all_excluded_features[idx],
                        prototypes=global_prototypes,
                        task_label_indices=task_label_indices,
                        num_labels=num_labels,
                    )
                    all_analysis_results[idx] = analysis

                    for res in analysis:
                        print(
                            f"    Matched label {res['matched_label']}: "
                            f"noise_ratio={res['noise_ratio']:.2%}, "
                            f"match_score={res['match_score']:.6f}, "
                            f"product={res['product']:.6f}"
                        )

            # -----------------------------------------------------------------
            # Phase 3 trigger: Relabeling (one round after analysis)
            # -----------------------------------------------------------------
            if (
                    self._cur_task > self.when_stage2added
                    and com == self.warmup_end + 1
                    and all_analysis_results
            ):
                print(f"\n[FedRNC] Phase 3: Relabeling")

                for idx in idxs_users:
                    if idx not in all_analysis_results:
                        continue

                    store = self._client_stores[idx]
                    analysis_results = all_analysis_results[idx]

                    # Collect all relabeling instructions (set of excluded indices)
                    relabel_indices = set()

                    for res in analysis_results:
                        matched_label = res["matched_label"]
                        product_val = res["product"]

                        if matched_label >= len(global_prototypes):
                            continue
                        proto = global_prototypes[matched_label]
                        if proto.norm() == 0:
                            continue

                        # Percentile from product value
                        percentile_value = min(max(product_val * 100, 0), 100)
                        if percentile_value < 1:
                            percentile_value = 25  # default

                        # Find noise-like samples in clean data
                        noiselike_clean = self._collect_noiselike_samples(
                            client_features=all_client_features[idx],
                            target_labels=task_label_indices,
                            target_prototype=proto,
                            percentile_value=percentile_value,
                        )

                        # Selective flipping on clean data
                        if noiselike_clean:
                            # Add matched label: 0 -> 1
                            self._relabel_samples(
                                store.data,
                                set(noiselike_clean),
                                matched_label,
                                1,
                            )
                            # Remove suspected noisy labels in task group: 1 -> 0
                            for tl in task_label_indices:
                                if tl != matched_label:
                                    self._relabel_samples(
                                        store.data,
                                        set(noiselike_clean),
                                        tl,
                                        0,
                                    )

                        # Find noise-like samples in excluded data
                        if len(store.excluded_data) > 0:
                            noiselike_excl = self._collect_noiselike_samples(
                                client_features=all_excluded_features[idx],
                                target_labels=task_label_indices,
                                target_prototype=proto,
                                percentile_value=percentile_value,
                            )

                            # Mark these excluded samples for relabeling
                            relabel_indices.update(noiselike_excl)

                    # Relabel excluded data: apply ALL matched labels first, then merge once
                    if relabel_indices and len(store.excluded_data) > 0:
                        for res in analysis_results:
                            matched_label = res["matched_label"]
                            store.relabel_samples(
                                relabel_map=relabel_indices,
                                matched_label=matched_label,
                                task_label_indices=task_label_indices,
                            )
                        # Merge relabeled excluded data back into clean data
                        store.merge_excluded()

                        print(
                            f"  Client {idx}: relabeled {len(relabel_indices)} samples, "
                            f"data size now: {len(store.data)}"
                        )

                    # Re-agg after relabeling
                    if self.args["num_users"] <= len(idxs_users):
                        # All clients participated - can re-aggregate
                        pass

            # Re-aggregate after relabeling round
            if (
                    self._cur_task > self.when_stage2added
                    and com == self.warmup_end + 1
                    and all_analysis_results
            ):
                local_weights_2 = []
                for idx in idxs_users:
                    store = self._client_stores[idx]
                    train_list = store.data
                    if self._cur_task > 0:
                        previous_dataset = self._get_previous_data_for_client(
                            data_manager, idx
                        )
                        if previous_dataset is not None and len(previous_dataset) > 0:
                            train_list = train_list + previous_dataset

                    local_train_loader = DataLoader(
                        train_list,
                        batch_size=self.args["local_bs"],
                        shuffle=True,
                        num_workers=self.args["num_worker"],
                        pin_memory=True,
                        multiprocessing_context=self.args["mulc"],
                        persistent_workers=False,
                    )
                    current_lr = scheduler.get_last_lr()[0]
                    w = self._local_finetune(
                        copy.deepcopy(self._network),
                        local_train_loader,
                        lr=current_lr,
                    )
                    local_weights_2.append(w)
                    del local_train_loader

                if local_weights_2:
                    global_weights = self._average_state_dicts(local_weights_2)
                    self._network.load_state_dict(global_weights)
                    del local_weights_2, global_weights

                del all_client_features, all_excluded_features, all_analysis_results, global_prototypes
                torch.cuda.empty_cache()

            self._save_checkpoint(com + 1, optimizer, scheduler)

        # Load best model
        self._network.load_state_dict(best_model)
        del best_model
        torch.cuda.empty_cache()

    def _get_previous_data_for_client(self, data_manager, client_idx):
        """
        Get previous-task replay data for a specific client (ensuring data privacy).
        Follows iCaRL's pattern with same-seed partitioning.
        """
        all_previous_data = []

        for task_idx in range(self._cur_task):
            task_datasets = data_manager.get_dataset(
                begin=task_idx, end=task_idx, mode="train"
            )

            for dataset in task_datasets:
                if not hasattr(dataset, "muti_labels"):
                    continue
                task_labels = dataset.muti_labels
                setup_seed(self.seed)
                user_groups, _ = partition_data(
                    task_labels,
                    beta=self.args["beta"],
                    n_parties=self.args["num_users"],
                )

                if client_idx not in user_groups:
                    continue

                client_dataset = DatasetSplit(dataset, user_groups[client_idx])

                # Sample replay data (memory budget per task)
                memory_per_task = self._memory_size // max(self._cur_task, 1)
                memory_per_client = memory_per_task // self.args["num_users"]

                for i in range(min(len(client_dataset), memory_per_client)):
                    image, label = client_dataset[i]
                    # Pad label to current total classes
                    if isinstance(label, torch.Tensor) and label.shape[0] < self._total_classes:
                        label = F.pad(
                            label,
                            (0, self._total_classes - label.shape[0]),
                            value=0.0,
                        )
                    all_previous_data.append((image, label))

        return all_previous_data if len(all_previous_data) > 0 else None
