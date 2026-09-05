import copy
import logging
import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from torchvision import transforms
import json

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args, logger):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 2
        self.args = args
        self.seed = args["seed"]
        self.select_img_seed = args["select_img_seed"]
        self.tasks = args["tasks"]
        self.wandb = args["wandb"]
        self.save_dir = args["save_dir"]
        self.dataset_names = args["dataset_names"]
        self.nums = args["nums"]
        self.logger = logger

        # ----
        args["memory_size"] = 1000
        # args["memory_per_class"] = 20
        args["fixed_memory"] = False

        self._memory_size = args["memory_size"]
        # self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = "0"
        # self._multiple_gpus = args["device"]

        self.classes = [
            "normal",
            "pneumonia",
            "covid",
            "enlarged cardiomediastinum",
            "cardiomegaly",
            "lung opacity",
            "lung lesion",
            "edema",
            "consolidation",
            "atelectasis",
            "pneumothorax",
            "pleural effusion",
            "pleural other",
            "fracture",
            "infiltration",
            "effusion",
            "nodule",
            "mass",
            "pleural_thickening",
            "emphysema",
            "fibrosis",
            "hernia"
        ]

    def _get_checkpoint_path(self):
        method_name = self.__class__.__name__.lower()
        if not hasattr(self, 'checkpoint_dir'):
            self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        return os.path.join(self.checkpoint_dir, f"{method_name}_task_{self._cur_task}.pth")

    def _save_checkpoint(self, com_round, optimizer, scheduler):
        if not hasattr(self, 'checkpoint_dir'):
            self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "cur_task": self._cur_task,
            "com_round": com_round,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "model_state": self._network.state_dict(),
            "best_model": getattr(self, 'best_model', getattr(self._network, 'state_dict', lambda: None)()),
            "best_val_auc": getattr(self, 'best_val_auc', 0),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
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
        if "best_model" in checkpoint and checkpoint["best_model"] is not None:
            self.best_model = checkpoint.get("best_model")
        else:
            self.best_model = copy.deepcopy(self._network.state_dict())
        self.best_val_auc = checkpoint.get("best_val_auc", 0)

        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] and optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        # if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] and scheduler:
        #     scheduler.load_state_dict(checkpoint["scheduler_state"])
        return checkpoint

    def after_task(self):
        pass

    def eval_task(self, data_manager):
        auc_per_class = {}
        avg_acc_per_dataset = []

        datasets = data_manager.get_dataset(begin=0, end=self._cur_task, mode="test")

        loaders = []
        for dataset in datasets:
            loader = DataLoader(
                dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"],
                multiprocessing_context=self.args["mulc"], persistent_workers=True
            )
            loaders.append(loader)

        for i, loader in enumerate(loaders):
            auc_dict, auc = self._single_compute_auc(self._network, loader)
            auc_per_class[self.dataset_names[i]] = auc_dict
            avg_acc_per_dataset.append(auc)

        overall_auc = np.round(np.mean(avg_acc_per_dataset), 4)
        info = f"Avg Per Dataset AUC: {avg_acc_per_dataset}, Overall AUC: {overall_auc}"
        self.logger.log(info)
        print(info)
        auc_path = f"{self.args['logdir']}/auc"
        os.makedirs(auc_path, exist_ok=True)
        result_path = f"{auc_path}/task_{self._cur_task}.json"
        with open(result_path, "w") as f:
            json.dump(auc_per_class, f, indent=2)
        self.logger.log(auc_per_class)
        print(auc_per_class)
        return overall_auc

    def _train(self):
        pass

    def _singal_compute_accuracy(self, model, loader):
        """
        Multi-label classification evaluation (CheXpert).
        Outputs per-class F1, macro F1, weighted F1, and per-class ROC-AUC.
        """
        threshold = self.args["threshold"]
        model.eval()
        y_true = []
        y_score = []  # Sigmoid probabilities used for AUC.

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.cuda()
                outputs = model(inputs)['logits']
                outputs = outputs[:, :targets.shape[1]]
                probs = torch.sigmoid(outputs)  # Multi-label outputs use sigmoid.

                y_true.append(targets.cpu().numpy())
                y_score.append(probs.cpu().numpy())

        # Merge all batches.
        y_true = np.concatenate(y_true, axis=0)
        y_score = np.concatenate(y_score, axis=0)

        # Binarize predictions for F1.
        y_pred = (y_score >= threshold).astype(int)

        # # Per-class F1.
        # f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Macro F1
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # # Micro F1 (optional).
        # micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # Weighted F1
        # print(y_true.shape, y_pred.shape)
        # weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class ROC-AUC.
        auc_per_class = []
        for i in range(len(y_true[0])):
            # If a class has one label value, AUC is undefined; use 0.5.
            if len(np.unique(y_true[:, i])) < 2:
                auc_per_class.append(0.5)
            else:
                auc_per_class.append(roc_auc_score(y_true[:, i], y_score[:, i]))

        return {
            # 'f1_per_class': np.round(f1_per_class, 4),
            'macro_f1': np.round(macro_f1, 4),
            # 'micro_f1': np.round(micro_f1, 4),
            # 'weighted_f1': np.round(weighted_f1, 4),
            'auc_per_class': np.round(auc_per_class, 4).tolist()
        }

    def _compute_accuracy(self, model, loaders):
        total_metrics = {
            'macro_f1': [],
            'auc_per_class': []
        }
        auc_per_class_list = []

        if isinstance(loaders, DataLoader):
            metrics = self._singal_compute_accuracy(model, loaders)
            total_metrics['macro_f1'].append(metrics['macro_f1'])
            auc_per_class_list.append(metrics['auc_per_class'])
        else:
            for loader in loaders:
                metrics = self._singal_compute_accuracy(model, loader)
                total_metrics['macro_f1'].append(metrics['macro_f1'])
                auc_per_class_list.append(metrics['auc_per_class'])

        # Compute mean AUC across classes.
        # avg_auc_per_class = np.mean(auc_per_class_list, axis=0) if auc_per_class_list else []

        # return {
        #     'weighted_f1': round(np.mean(total_metrics['weighted_f1']), 4),
        #     'auc_per_class': avg_auc_per_class
        # }
        return round(np.mean(total_metrics['macro_f1']), 4), auc_per_class_list

    def _single_compute_auc(self, model, loaders):
        """
        Compute ROC-AUC.
        """
        model.eval()
        y_true = []
        y_score = []

        with torch.no_grad():
            for inputs, targets in loaders:
                inputs = inputs.cuda()
                outputs = model(inputs)['logits']
                outputs = outputs[:, :targets.shape[1]]
                probs = torch.sigmoid(outputs)

                y_true.append(targets.cpu().numpy())
                y_score.append(probs.cpu().numpy())

        # Merge all batches.
        y_true = np.concatenate(y_true, axis=0)
        y_score = np.concatenate(y_score, axis=0)

        # Compute ROC-AUC for each class.
        auc_per_class = []
        for i in range(len(y_true[0])):
            if len(np.unique(y_true[:, i])) < 2:
                auc_per_class.append(None)
            else:
                auc_per_class.append(np.round(roc_auc_score(y_true[:, i], y_score[:, i]), 4))

        return {"AUC per class": {self.classes[i]: auc for i, auc in enumerate(auc_per_class)}, "AUC avg": np.round(
            np.mean([auc for auc in auc_per_class if auc is not None]), 4)}, np.round(
            np.mean([auc for auc in auc_per_class if auc is not None]), 4)

    def _compute_auc(self, model, loaders):
        total_auc = []

        if isinstance(loaders, DataLoader):
            _, auc = self._single_compute_auc(model, loaders)
            total_auc.append(auc)
        else:
            for loader in loaders:
                _, auc = self._single_compute_auc(model, loader)
                total_auc.append(auc)

        return round(np.mean(total_auc), 4)
