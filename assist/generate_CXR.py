import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch
import json

from methods.ours import ClipModel, MINIMGenerator, MINIMTrainer
from utils.data_manager import DataManager, partition_data, setup_seed

DEFAULT_DATA_PATH = [
    "/path/to/datasets/chest_xray",
    "/path/to/datasets/CXR_Covid-19_Challenge-001",
    "/path/to/datasets/CheXpert-v1.0-small",
    "/path/to/datasets/NIH_Chest_X_ray_14",
]

DEFAULT_DATA_NAME = ["CXR1", "CXR5", "CheXpert", "NIH14"]
DEFAULT_CLASSES = [2, 1, 11, 8]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated MINIM training + balanced multi-label combo generation per dataset"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="Generation GPUs, e.g. --gpus 0 1")
    parser.add_argument("--data_paths", type=str, nargs="+", default=DEFAULT_DATA_PATH)
    parser.add_argument("--dataset_names", type=str, nargs="+", default=DEFAULT_DATA_NAME)
    parser.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES)

    parser.add_argument("--num_users", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.0)

    parser.add_argument("--nums", type=int, nargs="+", help="Synthetic images per dataset")
    parser.add_argument("--output_dir", type=str, default="generated_cxr4")
    parser.add_argument(
        "--use_existing_model",
        action="store_true",
        help="If set, reuse existing MINIM model in output_dir/minim_task_{task_id} when available",
    )

    parser.add_argument(
        "--minim_pretrained_model_path",
        type=str,
        default="/path/to/models/RoentGen-v2",
    )
    parser.add_argument("--minim_train_batch_size", type=int, default=64)
    parser.add_argument("--minim_gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--minim_learning_rate", type=float, default=1e-5)
    parser.add_argument("--minim_com_rounds", type=int, default=5)
    parser.add_argument("--minim_local_epochs", type=int, default=1)
    parser.add_argument("--minim_frac", type=float, default=1.0)
    parser.add_argument("--minim_resolution", type=int, default=256)
    parser.add_argument("--minim_center_crop", action="store_true")
    parser.add_argument("--minim_random_flip", action="store_true")
    parser.add_argument("--minim_mixed_precision", type=str, default="fp16")
    parser.add_argument("--minim_noise_offset", type=float, default=0.0)
    parser.add_argument("--minim_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--minim_adam_beta1", type=float, default=0.9)
    parser.add_argument("--minim_adam_beta2", type=float, default=0.999)
    parser.add_argument("--minim_adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--minim_adam_epsilon", type=float, default=1e-8)

    parser.add_argument("--minim_generation_batch_size", type=int, default=8)
    parser.add_argument("--minim_num_inference_steps", type=int, default=50)
    parser.add_argument("--minim_guidance_scale", type=float, default=7.5)

    args = parser.parse_args()
    if not (
            len(args.data_paths) == len(args.dataset_names) == len(args.classes)
    ):
        raise ValueError("data_paths, dataset_names, and classes must have the same length")
    return args


def _generate_on_single_gpu(device_id, minim_model_path, pretrained_model_path, sampled_labels, batch_size,
                            num_inference_steps, guidance_scale):
    torch.cuda.set_device(device_id)
    clip_model = ClipModel(model_name="kaveh/rclip", device=f"cuda:{device_id}")
    generator = MINIMGenerator(
        minim_model_path=minim_model_path,
        pretrained_model_path=pretrained_model_path,
        device=f"cuda:{device_id}",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    images, labels, failed = generator.generate_from_labels(
        sampled_labels,
        clip_model,
        batch_size=batch_size,
    )

    # Return serializable objects from workers instead of passing PIL images.
    image_arrays = [np.asarray(img.convert("L"), dtype=np.uint8) for img in images]
    label_arrays = [
        lab.detach().cpu().numpy() if isinstance(lab, torch.Tensor) else np.asarray(lab)
        for lab in labels
    ]

    del generator
    torch.cuda.empty_cache()
    return image_arrays, label_arrays, failed


def _run_single_gpu_job(job):
    return _generate_on_single_gpu(*job)


def get_existing_minim_model_path(output_dir, task_id):
    """Return existing MINIM model path if valid, else None."""
    minim_model_path = os.path.join(output_dir, f"minim_task_{task_id}")
    unet_dir = os.path.join(minim_model_path, "unet")
    if os.path.isdir(unet_dir):
        return minim_model_path
    return None


def generate_multi_gpu(
        minim_model_path,
        pretrained_model_path,
        sampled_labels,
        gpu_ids,
        batch_size,
        num_inference_steps,
        guidance_scale,
):
    if len(sampled_labels) == 0:
        return [], [], []

    if len(gpu_ids) == 1:
        return _generate_on_single_gpu(
            gpu_ids[0],
            minim_model_path,
            pretrained_model_path,
            sampled_labels,
            batch_size,
            num_inference_steps,
            guidance_scale,
        )

    chunks = [[] for _ in gpu_ids]
    for i, lab in enumerate(sampled_labels):
        chunks[i % len(gpu_ids)].append(lab)

    # Pass only NumPy arrays and lists between processes to reduce pickle overhead.
    serializable_chunks = []
    for chunk in chunks:
        serializable_chunks.append([
            c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c)
            for c in chunk
        ])

    all_images, all_labels, all_failed = [], [], []
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(gpu_ids), mp_context=mp_ctx) as ex:
        futures = []
        for idx, device_id in enumerate(gpu_ids):
            if len(serializable_chunks[idx]) == 0:
                continue
            futures.append(
                ex.submit(
                    _run_single_gpu_job,
                    (
                        device_id,
                        minim_model_path,
                        pretrained_model_path,
                        serializable_chunks[idx],
                        batch_size,
                        num_inference_steps,
                        guidance_scale,
                    ),
                )
            )

        for fut in as_completed(futures):
            images, labels, failed = fut.result()
            all_images.extend(images)
            all_labels.extend(labels)
            all_failed.extend(failed)

    return all_images, all_labels, all_failed


def build_even_combo_labels(dataset, total_budget):
    """Distribute the budget across observed multi-label combinations."""
    if not hasattr(dataset, "muti_labels"):
        raise ValueError("Dataset missing muti_labels field")

    all_labels = dataset.muti_labels
    if isinstance(all_labels, torch.Tensor):
        all_labels = all_labels.cpu().numpy()
    all_labels = np.asarray(all_labels)

    if all_labels.size == 0:
        return [], {}

    combos = list(Counter(tuple(int(x) for x in row.tolist()) for row in all_labels).keys())
    if len(combos) == 0:
        return [], {}

    per_combo = total_budget // len(combos)
    remainder = total_budget % len(combos)

    sampled = []
    combo_targets = {}
    for i, combo in enumerate(combos):
        quota = per_combo + (1 if i < remainder else 0)
        combo_targets[combo] = quota
        for _ in range(quota):
            sampled.append(torch.tensor(combo, dtype=torch.float32))
    return sampled, combo_targets


def save_generated_dataset(images, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    label_vectors = []
    pseudo_vectors = []
    for idx, (img, label_vec) in enumerate(zip(images, labels)):
        save_path = os.path.join(out_dir, f"{idx:06d}.png")

        if isinstance(img, np.ndarray):
            from PIL import Image
            Image.fromarray(img).convert("L").save(save_path)
        else:
            img.convert("L").save(save_path)

        vec = label_vec.detach().cpu().numpy() if isinstance(label_vec, torch.Tensor) else np.asarray(label_vec)
        label_vectors.append(vec)
        # Pseudo-labels are the multi-label vectors used for generation prompts.
        # pseudo_vectors.append(vec)

    labels_np = np.asarray(label_vectors)
    # pseudos_np = np.asarray(pseudo_vectors)

    np.save(os.path.join(out_dir, "labels.npy"), labels_np)
    # np.save(os.path.join(out_dir, "pseudo_labels.npy"), pseudos_np)

    return len(images)


def main():
    args = parse_args()
    setup_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    data_manager = DataManager(args.data_paths, args.dataset_names, args.classes)

    minim_args = vars(args).copy()
    clip_model = ClipModel(model_name="kaveh/rclip", device=f"cuda:{args.gpu}")
    minim_trainer = MINIMTrainer(minim_args, clip_model=clip_model)

    gen_gpu_ids = args.gpus if args.gpus else [args.gpu]
    print(f"Generation GPUs: {gen_gpu_ids}")

    for task_id in range(data_manager.nb_tasks):
        train_dataset_list = data_manager.get_dataset(begin=task_id, end=task_id, mode="train")
        if len(train_dataset_list) == 0:
            print(f"[Task {task_id}] Empty train dataset, skip")
            continue

        train_dataset = train_dataset_list[0]
        if not hasattr(train_dataset, "muti_labels"):
            print(f"[Task {task_id}] Dataset has no muti_labels, skip")
            continue

        labels_np = train_dataset.muti_labels
        if isinstance(labels_np, torch.Tensor):
            labels_np = labels_np.cpu().numpy()

        user_groups, _ = partition_data(labels_np, beta=args.beta, n_parties=args.num_users)
        with open(os.path.join(args.output_dir, f"user_groups_task_{task_id}.json"), "w", encoding="utf-8") as fp:
            json.dump(user_groups, fp, indent=4)

        task_out_dir = os.path.join(args.output_dir, f"task_{task_id}")
        os.makedirs(task_out_dir, exist_ok=True)

        minim_model_path = None
        if args.use_existing_model:
            minim_model_path = get_existing_minim_model_path(args.output_dir, task_id)
            if minim_model_path is not None:
                print(f"\n[Task {task_id}] Found existing MINIM model at {minim_model_path}, skip training")
            else:
                print(f"\n[Task {task_id}] No existing MINIM model found, training from scratch")

        if minim_model_path is None:
            print(f"\n[Task {task_id}] Start federated MINIM training")
            minim_model_path = minim_trainer.train_minim_federated(
                train_dataset=train_dataset,
                output_dir=args.output_dir,
                task_id=task_id,
                user_groups=user_groups,
            )

        sampled_labels, combo_targets = build_even_combo_labels(train_dataset, total_budget=args.nums[task_id])
        if len(sampled_labels) == 0:
            print(f"[Task {task_id}] No valid multi-label combos, skip generation")
            continue

        print(
            f"[Task {task_id}] Generate {args.nums[task_id]} images by {len(combo_targets)} combos (even allocation)"
        )
        images, labels, failed = generate_multi_gpu(
            minim_model_path=minim_model_path,
            pretrained_model_path=args.minim_pretrained_model_path,
            sampled_labels=sampled_labels,
            gpu_ids=gen_gpu_ids,
            batch_size=args.minim_generation_batch_size,
            num_inference_steps=args.minim_num_inference_steps,
            guidance_scale=args.minim_guidance_scale,
        )
        saved_num = save_generated_dataset(images, labels, task_out_dir)

        with open(os.path.join(task_out_dir, "generation_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"task_id={task_id}\n")
            f.write(f"dataset={args.dataset_names[task_id]}\n")
            f.write(f"target_num={args.nums[task_id]}\n")
            f.write(f"saved_num={saved_num}\n")
            f.write(f"combo_count={len(combo_targets)}\n")
            f.write(f"failed_count={len(failed)}\n")

        print(f"[Task {task_id}] Saved {saved_num} images to {task_out_dir}")
        if len(failed) > 0:
            print(f"[Task {task_id}] failed labels: {len(failed)}")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
