import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb

lamda = 1000
fishermax = 0.0001


class EWC(BaseLearner):
    def __init__(self, args, logger=None):
        super().__init__(args, logger)
        # self.fisher = None
        self.fisher_list = {}
        self.mean_list = {}
        self._network = IncrementalNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_dataset = self.dataset_names[self._cur_task]
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        message = "Learning on {}: {}-{}".format(
            self._cur_dataset, self._known_classes, self._total_classes
        )
        print(message)
        if self.logger is not None:
            self.logger.log(message)

        train_dataset = data_manager.get_dataset(
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
            val_loader = DataLoader(
                val_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=self.args["num_worker"],
                multiprocessing_context=self.args["mulc"],
                persistent_workers=True,
            )
            self.val_loader.append(val_loader)

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

        if len(self.val_loader) == 0:
            self.val_loader = self.test_loader

        setup_seed(self.seed)

        train_set = train_dataset[0]
        train_labels = getattr(train_set, "muti_labels", None)
        if train_labels is None:
            train_labels = train_set.labels
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.cpu().numpy()
        user_groups, _ = partition_data(
            train_labels, beta=self.args["beta"], n_parties=self.args["num_users"]
        )

        self._fl_train(user_groups, train_set, self.val_loader, self.test_loader)

        # Save the model.
        if self._cur_task != self.tasks:
            torch.save(copy.deepcopy(self._network.state_dict()), 'store/{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                       .format(self.args["exp_name"], self._cur_dataset, self.args["net"], self.args["num_users"],
                               self.args["beta"], self.args["method"], self._cur_task, self.nums))

    def _local_update(self, model, train_data_loader, idx, lr):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args["weight_decay"]
        )
        for _ in range(self.args["local_ep"]):
            for batch in train_data_loader:
                images, labels = self._prepare_batch(batch)
                logits = model(images)["logits"]
                labels = self._prepare_labels(labels, logits.shape[1])
                logits = logits[:, : labels.shape[1]]
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.fisher_list[idx] = self.getFisherDiagonal(train_data_loader, model)
        self.mean_list[idx] = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        return model.state_dict()

    def _local_finetune(self, model, train_data_loader, idx, lr):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args["weight_decay"]
        )

        for _ in range(self.args["local_ep"]):
            for batch in train_data_loader:
                images, labels = self._prepare_batch(batch)
                logits = model(images)["logits"]
                labels = self._prepare_labels(labels, logits.shape[1])
                logits = logits[:, : labels.shape[1]]
                loss_clf = F.binary_cross_entropy_with_logits(logits, labels)
                loss_ewc = self.compute_ewc(model, idx)
                loss = loss_clf + lamda * loss_ewc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        alpha = self._known_classes / max(self._total_classes, 1)
        new_finsher = self.getFisherDiagonal(train_data_loader, model)
        if idx in self.fisher_list:
            for n, _ in new_finsher.items():
                if n in self.fisher_list[idx]:
                    new_finsher[n][: len(self.fisher_list[idx][n])] = (
                            alpha * self.fisher_list[idx][n]
                            + (1 - alpha) * new_finsher[n][: len(self.fisher_list[idx][n])]
                    )
        self.fisher_list[idx] = new_finsher
        self.mean_list[idx] = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        return model.state_dict()

    def _fl_train(self, user_groups, train_dataset, val_loader, test_loader):
        self._network.cuda()
        best_model = copy.deepcopy(self._network.state_dict())
        best_val_auc = float("-inf")

        total_rounds = self.args.get("com_rounds", None)
        if total_rounds is not None:
            total_rounds = self.args["com_rounds"][self._cur_task]
        else:
            total_rounds = self.args["com_round"]

        optimizer = torch.optim.SGD(
            self._network.parameters(),
            lr=self.args["local_lr"],
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

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

        prog_bar = tqdm(range(start_round, total_rounds))

        for com in prog_bar:
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                local_train_loader = DataLoader(
                    DatasetSplit(train_dataset, user_groups[idx]),
                    batch_size=self.args["local_bs"],
                    shuffle=True,
                    num_workers=self.args["num_worker"],
                    pin_memory=True,
                    multiprocessing_context=self.args["mulc"],
                    persistent_workers=True,
                )
                current_lr = scheduler.get_last_lr()[0]
                if self._cur_task == 0 or idx not in self.fisher_list:
                    w = self._local_update(
                        copy.deepcopy(self._network), local_train_loader, idx, current_lr
                    )
                else:
                    w = self._local_finetune(
                        copy.deepcopy(self._network), local_train_loader, idx, current_lr
                    )
                local_weights.append(copy.deepcopy(w))
                del local_train_loader
            scheduler.step()

            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            val_auc = self._compute_auc(copy.deepcopy(self._network), val_loader)
            print(f"Epoch {com}, Val AUC: {val_auc}")
            if val_auc >= best_val_auc:
                best_val_auc = val_auc
                best_model = copy.deepcopy(self._network.state_dict())
                self.best_model = best_model
                self.best_val_auc = best_val_auc

            self._save_checkpoint(com + 1, optimizer, scheduler)

        self._network.load_state_dict(best_model)

    def compute_ewc(self, model, idx):
        if idx not in self.fisher_list or idx not in self.mean_list:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher_list[idx]:
                loss += (
                        torch.sum(
                            (self.fisher_list[idx][n])
                            * (p[: len(self.mean_list[idx][n])] - self.mean_list[idx][n]).pow(2)
                        )
                        / 2
                )
        return loss

    def getFisherDiagonal(self, train_loader, model):
        fisher = {
            n: torch.zeros(p.shape).cuda()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        for batch in train_loader:
            images, targets = self._prepare_batch(batch)
            logits = model(images)["logits"]
            targets = self._prepare_labels(targets, logits.shape[1])
            logits = logits[:, : targets.shape[1]]
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax, device=fisher[n].device))
        return fisher

    def _prepare_batch(self, batch):
        if len(batch) == 3:
            _, images, labels = batch
        else:
            images, labels = batch
        return images.cuda(), labels.cuda()

    def _prepare_labels(self, labels, num_classes):
        if labels.dim() == 1:
            labels = F.one_hot(labels.long(), num_classes=num_classes).float()
        elif labels.dim() == 2 and labels.size(1) == 1:
            labels = F.one_hot(labels.squeeze(1).long(), num_classes=num_classes).float()
        else:
            labels = labels.float()
        return labels
