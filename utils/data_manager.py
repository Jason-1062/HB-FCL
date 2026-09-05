from torch.utils.data import Dataset
from dataset import CheXpertDataset, NIH_Chest_X_Ray_Dataset
from dataset.CXR_with_split_Dataset import CXR_with_split
from dataset.ChestXR_Dataset import ChestXR_Dataset
import torch, copy
import random
import numpy as np
import torch.backends.cudnn as cudnn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if "num_batches_tracked" in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def record_net_data_stats_multi_label(y_train, net_dataidx_map):
    """
    Record statistics for multi-label datasets.
    y_train: [n_samples, n_classes] binary label array.
    """
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        # Count positive samples per class and client.
        class_counts = {}
        client_labels = y_train[dataidx]
        for class_idx in range(client_labels.shape[1]):
            # Count samples with a positive label for this class.
            class_counts[class_idx] = np.sum(client_labels[:, class_idx])
        net_cls_counts[net_i] = class_counts

    print("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


def partition_data(y_train, beta=0.0, n_parties=5):
    """
    Non-IID partitioning for multi-label datasets with exclusive assignment.
    """
    y_train = np.array(y_train)
    n_samples = y_train.shape[0]
    n_classes = y_train.shape[1]

    net_dataidx_map = {}

    if beta == 0:  # IID
        idxs = np.random.permutation(n_samples)
        batch_idxs = np.array_split(idxs, n_parties)
        for i in range(n_parties):
            net_dataidx_map[i] = batch_idxs[i].tolist()

    else:  # Non-IID
        temp_labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Get active label indices for the sample.
            pos_labels = np.where(y_train[i] == 1)[0]
            if len(pos_labels) > 0:
                # Select one label at random as the partition key.
                temp_labels[i] = np.random.choice(pos_labels)
            else:
                # Edge case: assign unlabeled samples randomly.
                temp_labels[i] = np.random.randint(0, n_classes)

        min_size = 0
        min_require_size = 10

        idxs = np.arange(n_samples)

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]

            for k in range(n_classes):
                idx_k = idxs[temp_labels == k]
                if len(idx_k) == 0:
                    continue

                np.random.shuffle(idx_k)
                # Sample a Dirichlet distribution.
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))

                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                split_idx = np.split(idx_k, proportions)

                for j in range(n_parties):
                    idx_batch[j].extend(split_idx[j].tolist())

            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    train_data_cls_counts = record_net_data_stats_multi_label(y_train, net_dataidx_map)

    return net_dataidx_map, train_data_cls_counts


class DataManager(object):
    def __init__(self, data_paths, dataset_names=["ChestXR", "Knee"], classes=[3, 5]):
        self.dataset_names = dataset_names
        self._increments = classes
        self.data_paths = data_paths

        # Map dataset names to constructors.
        self.dataset_map = {
            # all CXR task
            "CXR": lambda split, index: CXR_with_split(data_paths[index], split=split),
            "ChestXR": lambda split, index: ChestXR_Dataset(
                data_paths[index], split=split
            ),
            "CheXpert": lambda split, index: CheXpertDataset(
                data_paths[index], split=split
            ),
            "NIH14": lambda split, index: NIH_Chest_X_Ray_Dataset(
                data_paths[index], split=split
            ),
        }

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, begin, end, mode="train"):
        """Return datasets for an inclusive range; labels are not aligned."""
        assert (
            0 <= begin <= end < len(self.dataset_names)
        ), f"Invalid begin-end range: {begin}-{end}, total datasets: {len(self.dataset_names)}"

        datasets = []
        for idx in range(begin, end + 1):
            dataset_name = self.dataset_names[idx]
            if dataset_name not in self.dataset_map:
                raise ValueError(f"Unknown dataset name: {dataset_name}")

            # Get the raw datasets.
            dataset = self.dataset_map[dataset_name](mode, idx)

            datasets.append(dataset)

        return datasets  # The returned list has unaligned labels.
