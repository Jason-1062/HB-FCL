import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import wandb, copy

T = 2
lamda = 3


class LwF(BaseLearner):
    def __init__(self, args, logger=None):
        super().__init__(args, logger)
        self._network = IncrementalNet(args, False)
        self.acc = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_dataset = self.dataset_names[self._cur_task] if hasattr(self,
                                                                          'dataset_names') else f"task{self._cur_task}"
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        print(f"Learning on {self._cur_dataset}: {self._known_classes}-{self._total_classes}")

        train_dataset = data_manager.get_dataset(begin=self._cur_task, end=self._cur_task, mode="train")
        val_datasets = data_manager.get_dataset(begin=0, end=self._cur_task, mode="val")
        test_datasets = data_manager.get_dataset(begin=0, end=self._cur_task, mode="test")

        self.val_loader = [
            DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=self.args.get("num_worker", 4)) for
            val_dataset in val_datasets]
        self.test_loader = [
            DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=self.args.get("num_worker", 4)) for
            test_dataset in test_datasets]
        if len(self.val_loader) == 0:
            self.val_loader = self.test_loader

        setup_seed(self.seed)
        train_set = train_dataset[0]
        train_labels = getattr(train_set, "muti_labels", None)
        if train_labels is None:
            train_labels = train_set.labels
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.cpu().numpy()
        user_groups, _ = partition_data(train_labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        self._fl_train(user_groups, train_set, self.val_loader, self.test_loader)

        # Save the model.
        if self._cur_task != self.tasks:
            torch.save(copy.deepcopy(self._network.state_dict()), 'store/{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                       .format(self.args["exp_name"], self._cur_dataset, self.args["net"], self.args["num_users"],
                               self.args["beta"], self.args["method"], self._cur_task, self.nums))

    def _local_update(self, model, train_data_loader, lr):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=self.args.get('weight_decay', 5e-4))
        for _ in range(self.args["local_ep"]):
            for batch_idx, (images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.binary_cross_entropy_with_logits(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def _local_finetune(self, model, train_data_loader, lr):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=self.args.get('weight_decay', 5e-4))
        for _ in range(self.args["local_ep"]):
            for batch_idx, (images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                # Multi-label classification loss.
                loss_clf = F.binary_cross_entropy_with_logits(output, labels)
                # Knowledge-distillation loss for previous-task classes.
                with torch.no_grad():
                    old_logits = self._old_network(images)["logits"]
                loss_kd = _KD_loss(output[:, :self._known_classes], old_logits, T)
                loss = lamda * loss_kd + loss_clf
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def _fl_train(self, user_groups, train_dataset, val_loader, test_loader):
        self._network.cuda()
        best_model = copy.deepcopy(self._network.state_dict())
        best_val_auc = float("-inf")
        total_rounds = self.args.get("com_rounds", None)
        if total_rounds is not None:
            total_rounds = total_rounds[self._cur_task]
        else:
            total_rounds = self.args["com_round"]

        prog_bar = tqdm(range(total_rounds))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            torch.optim.SGD(self._network.parameters(), lr=self.args.get('local_lr', 0.01)), milestones=[50, 75],
            gamma=0.1)
        for com in prog_bar:
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.args.get('local_lr',
                                                                                                            0.01)
            for idx in idxs_users:
                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]),
                                                batch_size=self.args["local_bs"], shuffle=True,
                                                num_workers=self.args.get("num_worker", 4))
                if self._cur_task == 0:
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader, lr=current_lr)
                else:
                    w = self._local_finetune(copy.deepcopy(self._network), local_train_loader, lr=current_lr)
                local_weights.append(copy.deepcopy(w))
                del local_train_loader
            scheduler.step()

            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            val_auc = self._compute_auc(self._network, val_loader)
            if val_auc >= best_val_auc:
                best_val_auc = val_auc
                best_model = copy.deepcopy(self._network.state_dict())

        self._network.load_state_dict(best_model)


def _KD_loss(pred, soft, T):
    return F.binary_cross_entropy_with_logits(pred / T, torch.sigmoid(soft / T)) * (T ** 2)
