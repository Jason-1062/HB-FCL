from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import copy
from methods.ours import average_weights, DatasetSplit


class MINIMGenerator:
    """Generate synthetic data with the MINIM diffusion model."""

    def __init__(
        self,
        minim_model_path,
        pretrained_model_path,
        device="cuda:0",
        num_inference_steps=50,
        guidance_scale=7.5,
        clip_similarity_threshold=0.2,
        max_regen_attempts=200,
    ):
        """
        Args:
            minim_model_path: Trained MINIM path containing the unet subdirectory.
            pretrained_model_path: Pretrained Stable Diffusion path.
            device: Compute device.
            num_inference_steps: Number of inference steps.
            guidance_scale: Classifier-free guidance scale.
            clip_similarity_threshold: Image-text CLIP cosine-similarity threshold.
            max_regen_attempts: Maximum regeneration attempts per label combination.
        """
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.clip_similarity_threshold = clip_similarity_threshold
        self.max_regen_attempts = max_regen_attempts

        # Load the MINIM model.
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(minim_model_path, "unet")
        )

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path, unet=unet, safety_checker=None
        ).to(device)

        self.pipeline.set_progress_bar_config(disable=True)

    def generate_from_labels(self, label_vectors, clip_model, batch_size=4):
        """
        Generate images from label vectors, preserving the supplied frequencies.

        Args:
            label_vectors: Label vectors or tensor of shape (N, num_classes).
            clip_model: ClipModel used to convert labels into text.
            batch_size: Generation batch size.

        Returns:
            images: List of PIL images.
            labels: Corresponding label vectors.
        """
        if isinstance(label_vectors, torch.Tensor):
            if len(label_vectors.shape) == 1:
                label_vectors = label_vectors.unsqueeze(0)
            label_vectors = label_vectors.cpu().numpy()
        else:
            label_vectors = [
                (
                    lv.detach().cpu().numpy()
                    if isinstance(lv, torch.Tensor)
                    else np.asarray(lv)
                )
                for lv in label_vectors
            ]

        from collections import Counter

        # Preserve externally supplied combination frequencies.
        label_freq = Counter(
            tuple(int(v) for v in (np.asarray(l) > 0.5).astype(np.int32).tolist())
            for l in label_vectors
        )
        total_to_generate = sum(label_freq.values())

        all_images = []
        all_label_vectors = []
        failed_labels_info = []

        self.pipeline.enable_attention_slicing()
        pbar = tqdm(total=total_to_generate, desc="Generating images", leave=False)

        for label_tuple, total_needed in label_freq.items():
            label_vec = torch.tensor(label_tuple, dtype=torch.float32)
            label_indices = clip_model.multilabel_to_indices(label_vec.unsqueeze(0))[0]
            prompt = clip_model.combine_texts_RoentGen(label_indices)

            generated = 0
            while generated < total_needed:
                cur_bs = min(batch_size, total_needed - generated)
                images = self.pipeline(
                    prompt=[prompt] * cur_bs,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                ).images

                for image in images:
                    image = image.convert("L")
                    all_images.append(image)
                    all_label_vectors.append(label_vec.clone())

                generated += len(images)
                pbar.update(len(images))
                torch.cuda.empty_cache()

        pbar.close()
        return all_images, all_label_vectors, failed_labels_info


class MINIMDatasetWrapper(torch.utils.data.Dataset):
    """Wrap a raw dataset for MINIM training without repeated augmentation."""

    def __init__(
        self,
        train_dataset,
        clip_model,
        tokenizer,
        resolution=256,
        center_crop=True,
        random_flip=True,
    ):
        self.train_dataset = train_dataset
        self.clip_model = clip_model
        self.tokenizer = tokenizer

        # Image preprocessing for MINIM.
        transform_list = [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR
            )
        ]
        transform_list.append(
            transforms.CenterCrop(resolution)
            if center_crop
            else transforms.RandomCrop(resolution)
        )
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.extend(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.transform = transforms.Compose(transform_list)

        self.original_dataset = train_dataset.dataset
        self.idxs = train_dataset.idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        real_idx = self.idxs[idx]
        image, label = self._load_raw_sample(real_idx)

        pixel_values = self.transform(image)

        if isinstance(label, torch.Tensor):
            label_vec = label.detach().cpu().numpy()
        else:
            label_vec = np.array(label)

        label_tensor = torch.tensor(label_vec, dtype=torch.float32).unsqueeze(0)
        label_indices = self.clip_model.multilabel_to_indices(label_tensor)[0]
        text = self.clip_model.combine_texts_RoentGen(label_indices)

        input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {"pixel_values": pixel_values, "input_ids": input_ids}

    def _load_raw_sample(self, index):
        """Load an unaugmented image and its labels."""
        ds = self.original_dataset

        if hasattr(ds, "image_paths"):
            img_path = ds.image_paths[index]
            raw_img = Image.open(img_path).convert("RGB")
            raw_label = ds.muti_labels[index]
            return raw_img, torch.as_tensor(raw_label, dtype=torch.float32)
        else:
            raise ValueError(
                "Dataset does not have image_paths attribute or index out of range."
            )


class MINIMTrainer:
    """Train the MINIM diffusion model with Accelerate and federated clients."""

    def __init__(self, args, clip_model=None):
        self.args = args
        self.accelerator = None
        self.clip_model = clip_model  # Reuse the supplied CLIP model.

    def _local_train_minim(
        self,
        unet,
        train_dataset,
        client_id,
        task_id,
        vae,
        text_encoder,
        tokenizer,
        noise_scheduler,
    ):
        """
        Train the MINIM model locally on one client.
        Auxiliary models are supplied externally to avoid repeated I/O.
        """
        device = f"cuda:{self.args['gpu']}"
        unet = unet.to(device)
        unet.train()

        # Optimizer.
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=self.args["minim_learning_rate"],
            betas=(
                self.args.get("minim_adam_beta1"),
                self.args.get("minim_adam_beta2"),
            ),
            weight_decay=self.args.get("minim_adam_weight_decay"),
            eps=self.args.get("minim_adam_epsilon"),
        )

        # Prepare training data.
        minim_dataset = MINIMDatasetWrapper(
            train_dataset=train_dataset,
            clip_model=self.clip_model,
            tokenizer=tokenizer,
            resolution=self.args.get("minim_resolution"),
            center_crop=self.args.get("minim_center_crop", True),
            random_flip=self.args.get("minim_random_flip", True),
        )

        if len(minim_dataset) == 0:
            raise ValueError(
                f"MINIM dataset is empty for client {client_id}, task {task_id}"
            )

        def collate_fn(batch):
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
            input_ids = torch.stack([item["input_ids"] for item in batch])
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        train_dataloader = DataLoader(
            minim_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.args["minim_train_batch_size"],
            num_workers=0,
            persistent_workers=False,
        )

        # Use mixed-precision training.
        use_autocast = self.args.get("minim_mixed_precision", "fp16") == "fp16"
        scaler = None
        if use_autocast:
            scaler = torch.cuda.amp.GradScaler()

        # Get gradient accumulation steps.
        grad_accum_steps = self.args.get("minim_gradient_accumulation_steps", 1)

        # Clear gradients before the first accumulation step.
        optimizer.zero_grad()

        for epoch in range(self.args.get("minim_local_epochs")):
            unet.train()
            for step, batch in enumerate(train_dataloader):

                # --- Data preparation ---
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(device).to(torch.float32)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                if self.args.get("minim_noise_offset", 0) > 0:
                    noise += self.args["minim_noise_offset"] * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[
                        0
                    ]

                # --- Forward pass ---
                if use_autocast:
                    with torch.cuda.amp.autocast():
                        model_pred = unet(
                            noisy_latents, timesteps, encoder_hidden_states
                        ).sample
                else:
                    model_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                # --- Loss computation ---
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Scale the loss by the accumulation steps.
                loss = loss / grad_accum_steps

                # --- Backward pass ---
                if use_autocast and scaler is not None:
                    scaler.scale(loss).backward()

                    # Update parameters at accumulation boundaries.
                    if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(
                        train_dataloader
                    ):
                        if self.args.get("minim_max_grad_norm", 1.0) > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                unet.parameters(),
                                self.args.get("minim_max_grad_norm", 1.0),
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()

                    # Update parameters at accumulation boundaries.
                    if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(
                        train_dataloader
                    ):
                        if self.args.get("minim_max_grad_norm", 1.0) > 0:
                            torch.nn.utils.clip_grad_norm_(
                                unet.parameters(),
                                self.args.get("minim_max_grad_norm", 1.0),
                            )
                        optimizer.step()
                        optimizer.zero_grad()

        # Release optimizer memory without deleting the supplied VAE or text encoder.
        del optimizer
        if use_autocast and scaler is not None:
            del scaler
        # torch.cuda.empty_cache()

        return unet.state_dict()

    def train_minim_federated(self, train_dataset, output_dir, task_id, user_groups):
        """
        Train MINIM with federated clients.
        Shared model loading is centralized.
        """
        print(
            f"Starting federated MINIM training: task {task_id}, clients={self.args['num_users']}"
        )
        device = f"cuda:{self.args['gpu']}"

        minim_output_dir = os.path.join(output_dir, f"minim_task_{task_id}")
        os.makedirs(minim_output_dir, exist_ok=True)

        # --- 1. Preload shared auxiliary models (VAE, text encoder, scheduler). ---
        print("Loading (VAE, TextEncoder, Scheduler)...")
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(
            self.args["minim_pretrained_model_path"], subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.args["minim_pretrained_model_path"], subfolder="text_encoder"
        ).to(device)
        vae = AutoencoderKL.from_pretrained(
            self.args["minim_pretrained_model_path"], subfolder="vae"
        ).to(device)
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.args["minim_pretrained_model_path"], subfolder="scheduler"
        )

        # Freeze parameters.
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)

        # Ensure ClipModel is initialized.
        if self.clip_model is None:
            raise ValueError("clip_model must be provided")
        self.clip_model._init_if_needed()

        # --- 2. Load the global UNet. ---
        global_unet = UNet2DConditionModel.from_pretrained(
            self.args["minim_pretrained_model_path"], subfolder="unet"
        )

        minim_com_rounds = self.args.get("minim_com_rounds")

        for com_round in range(minim_com_rounds):
            print(f"MINIM communication round {com_round + 1}/{minim_com_rounds}")

            m = max(int(self.args.get("minim_frac", 1.0) * self.args["num_users"]), 1)
            idxs_users = np.random.choice(
                range(self.args["num_users"]), m, replace=False
            )

            local_weights = []

            for idx in idxs_users:
                # print(f"  Client {idx} local MINIM training...")
                client_dataset = DatasetSplit(train_dataset, user_groups[idx])
                local_unet = copy.deepcopy(global_unet)

                # Pass the preloaded models.
                local_weights_dict = self._local_train_minim(
                    local_unet,
                    client_dataset,
                    idx,
                    task_id,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    noise_scheduler=noise_scheduler,
                )

                local_weights.append(copy.deepcopy(local_weights_dict))
                del local_unet
                # torch.cuda.empty_cache()

            global_weights = average_weights(local_weights)
            global_unet.load_state_dict(global_weights)

            print(
                f"MINIM round {com_round + 1} complete; aggregated {len(local_weights)} client models"
            )

        global_unet.save_pretrained(os.path.join(minim_output_dir, "unet"))
        print(f"Federated MINIM training complete; model saved to {minim_output_dir}")

        # Release shared models.
        del vae, text_encoder, tokenizer, noise_scheduler
        torch.cuda.empty_cache()

        return minim_output_dir
