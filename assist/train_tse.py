import os
import copy
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json
from utils.data_manager import DataManager, setup_seed, partition_data, DatasetSplit, average_weights
from utils.inc_net import IncrementalNet
from utils.clip_model import ClipModel
from methods.ours import AsymmetricLoss


def _compute_auc(model, loader, output_dim):
    """Calculates AUC for the specified output dimension."""
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.cuda()
            # ours.py convention: model returns a dict with 'logits'
            outputs = model(inputs)['logits']

            logits = outputs[:, :output_dim]
            probs = torch.sigmoid(logits)

            y_true.append(targets[:, :output_dim].cpu().numpy())
            y_score.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_score = np.concatenate(y_score, axis=0)

    auc_per_class = []
    for i in range(output_dim):
        if len(np.unique(y_true[:, i])) < 2:
            auc_per_class.append(None)  # Default for single-class
        else:
            auc_per_class.append(roc_auc_score(y_true[:, i], y_score[:, i]))

    avg_auc = np.mean([auc for auc in auc_per_class if auc is not None])
    return avg_auc, auc_per_class


asl = AsymmetricLoss()


def _local_update_tse(model, train_data_loader, lr, output_dim, atomic_embeddings, task_classes, r, tse_ltc, local_ep,
                      weight_decay):
    """Local client update for TSE (mirrors methods/ours._local_update_tse).

    Returns: (state_dict, total_loss, avg_loss_f)
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_f = 0.0

    for _ in range(local_ep):
        for images, labels in train_data_loader:
            images = images.cuda()
            labels = labels.cuda().float()
            output_list = model(images)
            logits = output_list['logits'][:, :output_dim]
            feature = output_list['att']

            label_tuples = [tuple(l.cpu().numpy()) for l in labels]
            target_f = torch.stack([atomic_embeddings[t].squeeze(0) for t in label_tuples], dim=0)
            target_f = target_f.to(feature.device)

            loss_f = torch.relu(torch.nn.functional.mse_loss(feature, target_f.detach()) - r)

            # cos = F.cosine_similarity(feature, target_f.detach(), dim=1)
            # loss_f = torch.relu(self.r - cos).mean()

            # loss_ce = F.binary_cross_entropy_with_logits(logits, labels)
            loss_ce = asl(logits, labels)
            loss = loss_ce + tse_ltc * loss_f

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            total_loss_ce += loss_ce.detach().item()
            total_loss_f += loss_f.detach().item()

    avg_loss = total_loss / len(train_data_loader)
    avg_loss_f = total_loss_f / len(train_data_loader)
    avg_loss_ce = total_loss_ce / len(train_data_loader)
    return model.state_dict(), avg_loss, avg_loss_ce, avg_loss_f


def main():
    parser = argparse.ArgumentParser(description='TSE (Text-guided Semantic Embedding) Training Script')
    parser.add_argument('--dataset_names', type=str, nargs='+', default=['CXR1', 'CXR5', 'CheXpert'])
    parser.add_argument('--classes', type=int, nargs='+', default=[2, 1, 11])
    parser.add_argument('--data_paths', type=str, nargs='+', default=[
        "/path/to/datasets/chest_xray",
        "/path/to/datasets/CXR_Covid-19_Challenge-001",
        "/path/to/datasets/CheXpert-v1.0-small"
    ])
    parser.add_argument('--net', type=str, default='densenet121')
    parser.add_argument('--tse_ltc', type=float, default=5, help='Weight for features alignment loss')
    parser.add_argument('--r', type=float, default=0.0, help='Tolerance radius; zero gives plain MSE loss')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--com_rounds', type=int, nargs='+', default=[30, 30, 30], help='Communication rounds per task')
    parser.add_argument('--frac', type=float, default=1.0, help='Fraction of clients per round')
    parser.add_argument('--num_users', type=int, default=5, help='Total simulated users')
    parser.add_argument('--local_ep', type=int, default=2, help='Local epochs per client')
    parser.add_argument('--local_bs', type=int, default=64, help='Local batch size for clients')
    parser.add_argument('--local_lr', type=float, default=1e-4, help='LR used in AdamW scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_rounds', type=int, default=None, help='Warmup rounds (int)')
    parser.add_argument('--save_dir', type=str, default='./tse_checkpoints')
    parser.add_argument('--clip_model', type=str, default="kaveh/rclip")
    parser.add_argument('--beta', type=float, default=0.0, help='Dirichlet partition beta for user split')
    args = parser.parse_args()

    # Example: CUDA_VISIBLE_DEVICES=1 python train_tse.py --dataset_names CheXpert --classes 14 --data_paths /path/to/datasets/CheXpert-v1.0-small --com_rounds 30 --save_dir ./tse_checkpoints/asl_new_chexpert

    assert len(args.dataset_names) == len(args.classes) == len(args.data_paths) == len(
        args.com_rounds), "Length of dataset_names, classes, data_paths, and com_rounds must match."

    setup_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    arg_dict = vars(args)
    json.dump(arg_dict, open(os.path.join(args.save_dir, "training_args.json"), "w"), indent=4)

    # 1. Load Data
    dm = DataManager(args.data_paths, args.dataset_names, args.classes)

    # 2. Setup Label Embeddings (using ClipModel to get embeddings for all classes)
    print(f"Initializing CLIP model: {args.clip_model}")
    clip = ClipModel(model_name=args.clip_model, device="cuda")

    # 3. Model & Optimizer
    # We will build model inside the task loop now.

    # 4. Training Loop for each task
    for task_idx in range(len(args.dataset_names)):
        print(f"\n{'=' * 20} Training TSE for Task {task_idx}: {args.dataset_names[task_idx]} {'=' * 20}")

        # Current task specific config
        task_classes = args.classes[task_idx]
        task_total_classes = sum(args.classes[:task_idx + 1])  # Important: ours.py TSEs predict all classes seen so far

        train_dataset = dm.get_dataset(task_idx, task_idx, mode='train')[0]
        val_dataset = dm.get_dataset(task_idx, task_idx, mode='val')[0]
        test_dataset = dm.get_dataset(task_idx, task_idx, mode='test')[0]

        task_val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)
        task_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)

        # Collect all labels.
        all_labels = train_dataset.muti_labels
        if isinstance(all_labels, torch.Tensor):
            all_labels = all_labels.cpu().numpy()

        # Build all unique label combinations.
        unique_label_tuples = list(set(tuple(l) for l in all_labels))

        # Compute CLIP embeddings in batches.
        label_embedding_cache = {}
        for label_tuple in unique_label_tuples:
            # Convert tuples to tensors before encoding.
            label_tensor = torch.tensor(label_tuple, dtype=torch.float32).unsqueeze(0)
            emb = clip.caculate_embdding(label_tensor)  # (1, emb_dim)
            label_embedding_cache[label_tuple] = emb.detach().cpu()

        print(f"Precomputed {len(unique_label_tuples)} unique label embeddings.")

        # Build model for current task (predicts all classes seen up to this task)
        model = IncrementalNet({'net': args.net, 'device': 'cuda'}, False, False)
        model.update_fc(task_total_classes)
        model = model.cuda()

        best_val_auc = -1.0
        final_test_auc = -1.0
        task_history = []

        # Federated training using communication rounds and SequentialLR scheduler
        total_rounds = args.com_rounds[task_idx]
        warmup_rounds = args.warmup_rounds if args.warmup_rounds is not None else int(0.1 * total_rounds)

        def warmup_lambda(round):
            if round < warmup_rounds:
                return (round + 1) / warmup_rounds
            else:
                return 1.0

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.local_lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_rounds - warmup_rounds, 1),
            eta_min=args.local_lr * 0.1
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_rounds]
        )

        # Partition data into simulated users
        user_groups, _ = partition_data(train_dataset.muti_labels, beta=args.beta, n_parties=args.num_users)

        prog_bar = tqdm(range(total_rounds), desc=f"TSE Federated Task {task_idx}")
        for com in prog_bar:
            local_weights = []
            loss_f_list = []
            loss_list = []
            loss_ce_list = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            for idx in idxs_users:
                local_train_loader = DataLoader(
                    DatasetSplit(train_dataset, user_groups[idx]),
                    batch_size=args.local_bs,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                )
                local_model = copy.deepcopy(model)
                w, loss, loss_ce, loss_f = _local_update_tse(
                    local_model,
                    local_train_loader,
                    scheduler.get_last_lr()[0],
                    task_total_classes,
                    label_embedding_cache,
                    task_classes,
                    args.r,
                    args.tse_ltc,
                    args.local_ep,
                    args.weight_decay,
                )
                print(f"Round {com + 1}, User {idx}: loss={loss:.6f}, loss_ce={loss_ce:.6f}, loss_f={loss_f:.6f}")
                local_weights.append(copy.deepcopy(w))
                loss_f_list.append(loss_f)
                loss_ce_list.append(loss_ce)
                loss_list.append(loss)

            # Aggregate
            global_weights = average_weights(local_weights)
            model.load_state_dict(global_weights)
            scheduler.step()

            # Validation
            val_auc, _ = _compute_auc(model, task_val_loader, task_total_classes)
            test_auc, _ = _compute_auc(model, task_test_loader, task_total_classes)
            mean_loss_f = float(np.mean(loss_f_list)) if len(loss_f_list) > 0 else 0.0
            mean_loss = float(np.mean(loss_list)) if len(loss_list) > 0 else 0.0
            mean_loss_ce = float(np.mean(loss_ce_list)) if len(loss_ce_list) > 0 else 0.0
            prog_bar.set_postfix(
                {'val_auc': f"{val_auc:.4f}", 'test_auc': f"{test_auc:.4f}", 'loss_f': f"{mean_loss_f:.6f}",
                 'loss': f"{mean_loss:.6f}", 'loss_ce': f"{mean_loss_ce:.6f}"})

            epoch_info = {
                'round': com + 1,
                'val_auc': round(val_auc, 4),
                'test_auc': round(test_auc, 4),
                'loss_f': mean_loss_f,
                'loss': mean_loss,
                'loss_ce': mean_loss_ce
            }
            task_history.append(epoch_info)

            # append to consolidated txt log
            log_path = os.path.join(args.save_dir, "tse_training_log.txt")
            with open(log_path, "a") as lf:
                lf.write(
                    f"task={task_idx} round={com + 1} val_auc={val_auc:.4f} test_auc={test_auc:.4f} loss_f={mean_loss_f:.6f} loss={mean_loss:.6f} loss_ce={mean_loss_ce:.6f}\n")

            if val_auc >= best_val_auc:
                best_val_auc = val_auc
                final_test_auc = test_auc
                torch.save({
                    'model_state': model.state_dict(),
                    'output_dim': task_total_classes,
                }, os.path.join(args.save_dir, f"tse_task_{task_idx}.pth"))
                print(f"Best model for Task {task_idx} saved (format aligned with ours.py)")

        # Append task summary to consolidated txt log
        summary_path = os.path.join(args.save_dir, "tse_training_log.txt")
        with open(summary_path, "a") as lf:
            lf.write(
                f"TASK_SUMMARY task={task_idx} dataset={args.dataset_names[task_idx]} best_val_auc={best_val_auc:.4f} rounds={total_rounds} final_test_auc={final_test_auc:.4f}\n")


if __name__ == "__main__":
    main()
