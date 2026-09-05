import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb
import os
from sklearn.metrics import confusion_matrix


class Finetune(BaseLearner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self._network = IncrementalNet(args, False)
        self.acc = []
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_checkpoint_path(self):
        return os.path.join(self.checkpoint_dir, f"finetune_task_{self._cur_task}.pth")

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
        # print(f"Trying to load checkpoint: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            return None
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if checkpoint.get("cur_task") != self._cur_task:
            return None
        # print(f"Loaded checkpoint: task {checkpoint['cur_task']}, round {checkpoint['com_round']}")
        self._network.load_state_dict(checkpoint["model_state"])
        self.best_model = checkpoint.get(
            "best_model", copy.deepcopy(self._network.state_dict())
        )
        self.best_val_auc = checkpoint.get("best_val_auc", float("-inf"))
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        return checkpoint

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_dataset = self.dataset_names[self._cur_task]
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        # torch.save(self._network.state_dict(), 'init_model.pkl')
        print(
            "Learning on {}: {}-{}".format(
                self._cur_dataset, self._known_classes, self._total_classes
            )
        )

        train_dataset = data_manager.get_dataset(
            begin=self._cur_task, end=self._cur_task, mode="train"
        )
        val_datasets = data_manager.get_dataset(begin=0, end=self._cur_task, mode="val")
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

    def _local_update(self, model, train_data_loader, lr):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )
        loss_list = []
        for iter in range(self.args["local_ep"]):
            total_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.binary_cross_entropy_with_logits(output, labels)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_list.append(loss.item())
            print(
                f"Epoch {iter + 1}/{self.args['local_ep']}, Loss: {total_loss / len(train_data_loader):.4f}"
            )
        return model.state_dict(), np.mean(loss_list)

    def _local_finetune(self, model, train_data_loader, lr):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )
        # print_data_stats(0, train_data_loader)
        loss_list = []
        for iter in range(self.args["local_ep"]):
            total_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                # fake_targets = labels - self._known_classes
                output = model(images)["logits"]
                # * finetune on the new tasks
                loss = F.binary_cross_entropy_with_logits(output, labels)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_list.append(loss.item())
            print(
                f"Epoch {iter + 1}/{self.args['local_ep']}, Loss: {total_loss / len(train_data_loader):.4f}"
            )

        return model.state_dict(), np.mean(loss_list)

    def _fl_train(self, user_groups, train_dataset, val_loader, test_loader):
        self._network.cuda()
        self.best_model = copy.deepcopy(self._network.state_dict())
        self.best_val_auc = float("-inf")
        total_rounds = self.args.get("com_rounds", None)
        if total_rounds is not None:
            total_rounds = total_rounds[self._cur_task]
        else:
            total_rounds = self.args["com_round"]

        optimizer = torch.optim.SGD(
            self._network.parameters(),
            lr=self.args["local_lr"],
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )

        start_round = 0
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[50, 75], gamma=0.1
        )
        checkpoint = self._load_checkpoint(optimizer, scheduler)
        if checkpoint is not None:
            start_round = int(checkpoint.get("com_round", 0))
            print(
                f"Resuming from checkpoint: task {self._cur_task}, round {start_round}/{total_rounds}"
            )
            self.logger.log(
                f"Resume checkpoint: task {self._cur_task}, com_round {start_round}/{total_rounds}"
            )

        if start_round >= total_rounds:
            self._network.load_state_dict(self.best_model)
            torch.cuda.empty_cache()
            return True

        prog_bar = tqdm(range(start_round, total_rounds))
        for com in prog_bar:
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(
                range(self.args["num_users"]), m, replace=False
            )

            loss_list = []

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
                if self._cur_task == 0:
                    w, loss = self._local_update(
                        copy.deepcopy(self._network), local_train_loader, lr=current_lr
                    )
                else:
                    w, loss = self._local_finetune(
                        copy.deepcopy(self._network), local_train_loader, lr=current_lr
                    )
                local_weights.append(copy.deepcopy(w))
                loss_list.append(loss.item())
                del local_train_loader
            scheduler.step()

            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            val_auc = self._compute_auc(copy.deepcopy(self._network), val_loader)
            print(f"Epoch {com}, Val AUC: {val_auc}")
            if val_auc >= self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_model = copy.deepcopy(self._network.state_dict())

            test_auc = self._compute_auc(copy.deepcopy(self._network), test_loader)
            self.logger.log(
                f"lr: {scheduler.get_last_lr()[0]:.6f},"
                f"Task {self._cur_task}, Epoch {com + 1}/{total_rounds} => Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}, Loss: {np.mean(loss_list):.4f}"
            )

            if self._cur_task > 0:
                old_auc = self._compute_auc(
                    copy.deepcopy(self._network), self.old_loader
                )
                new_auc = self._compute_auc(
                    copy.deepcopy(self._network), self.new_loader
                )
                self.logger.log(
                    f"Task {self._cur_task}, Epoch {com + 1}/{total_rounds} => Old AUC: {old_auc:.4f}, New AUC: {new_auc:.4f}\n"
                )

            self._save_checkpoint(com + 1, optimizer, scheduler)

        self._network.load_state_dict(self.best_model)
        del self.best_model
        torch.cuda.empty_cache()
        return False
