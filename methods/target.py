import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb
from abc import ABC
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from kornia import augmentation
import time, os, math
import torch.nn.init as init
from PIL import Image

# Use the ImageNet-100 configuration while generating medical CXR images.
syn_round = 16
warmup = 20

synthesis_batch_size = 64
sample_batch_size = 64
g_steps = 50
is_maml = 0
kd_steps = 400
lr_g = 0.0002
lr_z = 0.01
oh = 0.1
T = 5
act = 0.0
adv = 1.0
bn = 0.1
reset_l0 = 0
reset_bn = 0
bn_mmt = 0.9
tau = 1
kd = 0.05
# Normalization parameters for grayscale images.
data_normalize = dict(mean=[0.5], std=[0.25])


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


# The normalizer is created from the dataset at runtime.


def _collect_all_images(nums, root, postfix=["png", "jpg", "jpeg", "JPEG"]):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            if nums != None:
                files.sort()
                # random.shuffle(files)
                files = files[:nums]
                # files = files[20*256:20*256+nums]       # discard the ealry-stage data
                # files = files[-nums:]  # 40*256
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, self.root)
        self.transform = transform

        # Try to load the label file with multi-label support.
        labels_file = os.path.join(self.root, "labels.npy")
        if os.path.exists(labels_file):
            self.labels = np.load(labels_file)
            # Ensure image and label counts match.
            if len(self.labels) != len(self.images):
                print(
                    f"Warning: Label count ({len(self.labels)}) != Image count ({len(self.images)}) in {self.root}"
                )
                min_len = min(len(self.labels), len(self.images))
                self.labels = self.labels[:min_len]
                self.images = self.images[:min_len]
        else:
            self.labels = None

    def __getitem__(self, idx):
        with Image.open(self.images[idx]) as img:
            img = img.copy()
        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = torch.from_numpy(self.labels[idx]).float()
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        has_labels = "with labels" if self.labels is not None else "without labels"
        return (
            "Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s\n\t%s"
            % (self.root, len(self), self.transform, has_labels)
        )


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0, 3, 1, 2)  # make it channel first
    assert len(images.shape) == 4
    assert isinstance(images, np.ndarray)

    N, C, H, W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros(
        (C, H * row + padding * (row - 1), W * col + padding * (col - 1)),
        dtype=images.dtype,
    )
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h : h + H, w : w + W] = img
    return pack


def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67)  # , alpha=40


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)  # , alpha=0.67


def reset_l0_fun(model):
    for n, m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    """
    Save a batch of grayscale images in 244x244x1 format.
    """
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    base_dir = os.path.dirname(output)
    if base_dir != "":
        os.makedirs(base_dir, exist_ok=True)

    def _to_image_array(img_data):
        """Convert channel-first/last arrays to Image-compatible format."""
        if img_data.ndim == 2:
            return img_data, "L"
        if img_data.ndim == 3:
            if img_data.shape[0] in (1, 3):
                if img_data.shape[0] == 1:
                    return img_data[0], "L"
                return np.transpose(img_data, (1, 2, 0)), "RGB"
            if img_data.shape[-1] in (1, 3):
                if img_data.shape[-1] == 1:
                    return img_data[..., 0], "L"
                return img_data, "RGB"
        raise ValueError(f"Unsupported image shape for saving: {img_data.shape}")

    if pack:
        imgs = pack_images(imgs, col=col)
        img_array, mode = _to_image_array(imgs)
        imgs = Image.fromarray(img_array, mode=mode)
        if size is not None:
            if isinstance(size, (list, tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                scale = size / max(h, w)
                imgs = imgs.resize([int(w * scale), int(h * scale)])
        imgs.save(output)
    else:
        output_filename = output.rsplit(".", 1)[0]  # Remove the file extension.
        for idx, img_data in enumerate(imgs):
            img_array, mode = _to_image_array(img_data)
            img = Image.fromarray(img_array, mode=mode)
            img.save(f"{output_filename}-{idx}.png")


class DeepInversionHook:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = (
            input[0]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view([nch, -1])
            .var(1, unbiased=False)
        )
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
                module.running_mean.data - mean, 2
            )
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(
                module.running_var.data
                - (1 - self.mmt_rate) * var
                - self.mmt_rate * var_mmt,
                2,
            ) + torch.norm(
                module.running_mean.data
                - (1 - self.mmt_rate) * mean
                - self.mmt_rate * mean_mmt,
                2,
            )

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (
                self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data,
            )

    def remove(self):
        self.hook.remove()


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0
        self._labels = []  # Store labels.

    def add(self, imgs, targets=None):
        save_image_batch(
            imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False
        )
        if targets is not None:
            # Save labels.
            if isinstance(targets, torch.Tensor):
                targets_np = targets.cpu().numpy()
            else:
                targets_np = np.array(targets)
            self._labels.append(targets_np)
        self._idx += 1

    def save_labels(self):
        """Save all labels to labels.npy."""
        if len(self._labels) > 0:
            all_labels = np.concatenate(self._labels, axis=0)
            labels_file = os.path.join(self.root, "labels.npy")
            np.save(labels_file, all_labels)
            print(f"Saved {len(all_labels)} labels to {labels_file}")

    def get_dataset(self, nums=None, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform, nums=nums)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=244, nc=1):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # return a copy of its own
    def clone(self):
        clone = Generator(
            self.params[0], self.params[1], self.params[2], self.params[3]
        )
        clone.load_state_dict(self.state_dict())
        return clone.cuda()


def kldiv(logits, targets, T=1.0, reduction="batchmean", eps=1e-8):
    # student: q, teacher: p
    z_s = logits / T
    z_t = targets / T

    p = torch.sigmoid(z_t).clamp(min=eps, max=1.0 - eps)  # teacher prob
    log_q = F.logsigmoid(z_s)  # log q
    log_1_q = F.logsigmoid(-z_s)  # log (1-q)

    # KL(Bernoulli(p) || Bernoulli(q)) per class
    kl = p * (torch.log(p) - log_q) + (1.0 - p) * (torch.log1p(-p) - log_1_q)

    if reduction == "none":
        out = kl
    elif reduction == "sum":
        out = kl.sum()
    elif reduction == "mean":
        out = kl.mean()
    elif reduction == "batchmean":
        out = kl.sum() / kl.size(0)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    return out * (T * T)


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction="batchmean"):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


class GlobalSynthesizer(ABC):
    def __init__(
        self,
        teacher,
        student,
        generator,
        nz,
        num_classes,
        img_size,
        init_dataset=None,
        iterations=100,
        lr_g=0.1,
        synthesis_batch_size=128,
        sample_batch_size=128,
        adv=0.0,
        bn=1,
        oh=1,
        save_dir="run/fast",
        transform=None,
        autocast=None,
        use_fp16=False,
        normalizer=None,
        distributed=False,
        lr_z=0.01,
        warmup=10,
        reset_l0=0,
        reset_bn=0,
        bn_mmt=0,
        is_maml=1,
        args=None,
    ):
        super(GlobalSynthesizer, self).__init__()
        self.teacher = teacher
        self.student = student
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.ismaml = is_maml
        self.args = args

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.normalizer = normalizer

        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.generator = generator.cuda().train()
        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None

        if self.ismaml:
            self.meta_optimizer = torch.optim.Adam(
                self.generator.parameters(),
                self.lr_g * self.iterations,
                betas=[0.5, 0.999],
            )
        else:
            self.meta_optimizer = torch.optim.Adam(
                self.generator.parameters(),
                self.lr_g * self.iterations,
                betas=[0.5, 0.999],
            )

        # Augmentation for grayscale images.
        # Convert grayscale images to three channels to match model input.
        self.aug = transforms.Compose(
            [
                augmentation.RandomCrop(
                    size=[self.img_size[-2], self.img_size[-1]], padding=4
                ),
                # augmentation.RandomHorizontalFlip(),
                normalizer,
            ]
        )

        self.bn_mmt = bn_mmt
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

    def synthesize(self, targets=None):
        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        if (self.ep == 120 + self.ep_start) and self.reset_l0:
            reset_l0_fun(self.generator)

        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda()
        z.requires_grad = True

        assert targets is not None, "Multi-label targets must be provided"

        targets = targets.cuda()

        fast_generator = self.generator.clone()
        optimizer = torch.optim.Adam(
            [{"params": fast_generator.parameters()}, {"params": [z], "lr": self.lr_z}],
            lr=self.lr_g,
            betas=[0.5, 0.999],
        )
        for it in range(self.iterations):
            inputs = fast_generator(z)
            # Replicate the grayscale channel to match model input.
            if inputs.shape[1] == 1:
                inputs = inputs.repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
            inputs_aug = self.aug(inputs)  # crop and normalize
            if it == 0:
                originalMeta = inputs
            t_out = self.teacher(inputs_aug)["logits"]
            # For multi-label data, derive targets from outputs when targets is None.
            if targets is None:
                targets = (torch.sigmoid(t_out) > 0.5).float()
                targets = targets.cuda()

            loss_bn = sum([h.r_feature for h in self.hooks])
            # Multi-label loss uses binary_cross_entropy_with_logits.
            loss_oh = F.binary_cross_entropy_with_logits(t_out, targets)
            if self.adv > 0 and (self.ep >= self.ep_start):
                s_out = self.student(inputs_aug)["logits"]
                # Compare sigmoid probabilities for multi-label outputs.
                s_pred = (torch.sigmoid(s_out) > 0.5).float()
                t_pred = (torch.sigmoid(t_out) > 0.5).float()
                mask = (s_pred == t_pred).all(dim=1).float()
                # Use binary_cross_entropy_with_logits for adversarial distillation.
                kd_loss = F.binary_cross_entropy_with_logits(
                    s_out, torch.sigmoid(t_out.detach()), reduction="none"
                )
                loss_adv = -(kd_loss.sum(1) * mask).mean()  # adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data.cpu()  # mem, cpu

            optimizer.zero_grad()
            loss.backward()

            if self.ismaml:
                if it == 0:
                    self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations - 1):
                    self.meta_optimizer.step()

            optimizer.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # REPTILE meta gradient
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            self.meta_optimizer.step()

        self.student.train()
        self.prev_z = (z, targets)
        end = time.time()

        self.data_pool.add(best_inputs, targets)  # add a batch of data with labels


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).cuda()
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits


class TARGET(BaseLearner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self._network = IncrementalNet(args, False, False)
        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        test_f1, test_auc = self._compute_accuracy(self._old_network, self.test_loader)
        print("After Test F1: %s" % test_f1)
        print("After Test AUC: %s" % test_auc)
        if hasattr(self, "logger"):
            self.logger.log("After Test F1: %s" % test_f1)
            self.logger.log("After Test AUC: %s" % test_auc)

    def kd_train(self, student, teacher, criterion, optimizer):
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data()
        data_iter = DataIter(loader)
        for i in range(kd_steps):
            data = data_iter.next()
            if torch.is_tensor(data):
                images = data
            elif isinstance(data, tuple):
                images = data[0]
            elif isinstance(data, list):
                if len(data) == 2 and torch.is_tensor(data[0]):
                    images = data[0]
                elif all(torch.is_tensor(item) for item in data):
                    images = torch.stack(data, dim=0)
                else:
                    raise TypeError(
                        f"Unsupported data format returned by loader: {type(data)}"
                    )
            else:
                raise TypeError(f"Unsupported data type from loader: {type(data)}")

            if isinstance(images, list):
                if all(torch.is_tensor(item) for item in images):
                    images = torch.stack(images, dim=0)
                elif all(isinstance(item, np.ndarray) for item in images):
                    images = torch.from_numpy(np.stack(images, axis=0))
                else:
                    raise TypeError(
                        "Loader returned a list with non-array elements, cannot stack."
                    )
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)
            if not torch.is_tensor(images):
                raise TypeError(
                    f"Unable to convert loader output to tensor, got {type(images)}"
                )

            images = images.cuda()
            with torch.no_grad():
                t_out = teacher(images)["logits"]
            s_out = student(images.detach())["logits"]
            # Use binary_cross_entropy_with_logits for multi-label distillation.
            loss_s = F.binary_cross_entropy_with_logits(
                s_out, torch.sigmoid(t_out.detach())
            )
            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()

    def _get_pre_task_label_distribution(self, data_manager):
        current_datasets = data_manager.get_dataset(
            begin=0, end=self._cur_task, mode="train"
        )

        all_labels = []
        for dataset in current_datasets:
            if hasattr(dataset, "muti_labels"):
                labels = dataset.muti_labels
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                all_labels.extend(labels)

        # Count occurrences of each label combination.
        from collections import Counter

        label_counts = Counter(tuple(l) for l in all_labels)
        total_real_samples = sum(label_counts.values())

        label_keys = list(label_counts.keys())
        print(
            f"Task {self._cur_task}: {len(label_counts)} unique label combinations, {total_real_samples} real samples"
        )
        return label_keys

    def data_generation(self, data_manager):
        nz = 256
        img_size = 244  # Medical CXR image size.
        img_shape = (1, 244, 244)  # Grayscale image shape.

        # Create transforms and a normalizer for grayscale images.
        # The model expects three channels, so grayscale input is replicated.
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop(244, scale=(0.8, 1.0)),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.Grayscale(
                    num_output_channels=3
                ),  # Replicate grayscale input.
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        data_normalize_3ch = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalizer = Normalizer(**data_normalize_3ch)

        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=1).cuda()
        student = copy.deepcopy(self._network)
        student.apply(weight_init)
        tmp_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        synthesizer = GlobalSynthesizer(
            copy.deepcopy(self._network),
            student,
            generator,
            nz=nz,
            num_classes=self._total_classes,
            img_size=img_shape,
            init_dataset=None,
            save_dir=tmp_dir,
            transform=train_transform,
            normalizer=normalizer,
            synthesis_batch_size=synthesis_batch_size,
            sample_batch_size=sample_batch_size,
            iterations=g_steps,
            warmup=warmup,
            lr_g=lr_g,
            lr_z=lr_z,
            adv=adv,
            bn=bn,
            oh=oh,
            reset_l0=reset_l0,
            reset_bn=reset_bn,
            bn_mmt=bn_mmt,
            is_maml=is_maml,
            args=self.args,
        )

        optimizer = torch.optim.SGD(
            student.parameters(), lr=0.2, weight_decay=0.0001, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 200, eta_min=2e-4
        )

        label_keys = self._get_pre_task_label_distribution(data_manager)

        for it in range(syn_round):
            # Sample batch_size labels from label_keys and concatenate them into targets.
            if label_keys and len(label_keys) > 0:
                indices = np.random.choice(
                    len(label_keys), size=synthesis_batch_size, replace=True
                )
                sampled_labels = [label_keys[i] for i in indices]
                targets = torch.stack(
                    [
                        F.pad(
                            torch.tensor(l, dtype=torch.float32),
                            (0, self._total_classes - len(l)),
                            mode="constant",
                            value=0.0,
                        )
                        for l in sampled_labels
                    ]
                ).cuda()
            else:
                targets = None

            synthesizer.synthesize(targets=targets)  # generate synthetic data
            print(
                "Task {}, Data Generation, Epoch {}/{} =>  Generating synthetic data...".format(
                    self._cur_task,
                    it + 1,
                    syn_round,
                )
            )

            if it >= warmup:
                self.kd_train(student, self._network, None, optimizer)  # kd_steps
                test_f1, test_auc = self._compute_accuracy(student, self.test_loader)
                print(
                    "Task {}, Data Generation, Epoch {}/{} =>  Student test_f1: {:.4f}".format(
                        self._cur_task,
                        it + 1,
                        syn_round,
                        test_f1,
                    )
                )
                scheduler.step()

        # Save the label file.
        synthesizer.data_pool.save_labels()

        print("For task {}, data generation completed! ".format(self._cur_task))

    def get_syn_data_loader(self):
        syn_bs = self.args["syn_bs"] * self.args["local_bs"]
        task_idx = self._cur_task - 1
        data_dir = os.path.join(self.save_dir, f"task_{task_idx}")
        if not os.path.exists(data_dir):
            return None
        syn_dataset = UnlabeledImageDataset(
            data_dir, transform=self.transform, nums=self.nums
        )
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset,
            batch_size=syn_bs,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.args["num_worker"],
            multiprocessing_context=self.args["mulc"],
        )
        return syn_data_loader

    def get_all_syn_data(self):
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        syn_dataset = UnlabeledImageDataset(
            data_dir, transform=self.transform, nums=self.args.get("nums", None)
        )
        loader = torch.utils.data.DataLoader(
            syn_dataset,
            batch_size=sample_batch_size,
            shuffle=True,
            num_workers=self.args.get("num_worker", 4),
            pin_memory=True,
            sampler=None,
        )
        return loader

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_dataset = self.dataset_names[self._cur_task]
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        # Create transforms and a normalizer for grayscale images.
        # Replicate grayscale input because the model expects three channels.
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop(244, scale=(0.8, 1.0)),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.Grayscale(
                    num_output_channels=3
                ),  # Replicate grayscale input.
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        data_normalize_3ch = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalizer = Normalizer(**data_normalize_3ch)

        self._network.update_fc(self._total_classes)
        message = "Learning on {}: {}-{}".format(
            self._cur_dataset, self._known_classes, self._total_classes
        )
        print(message)
        if hasattr(self, "logger"):
            self.logger.log(message)

        train_dataset = data_manager.get_dataset(  # * get the data for one task
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
                num_workers=self.args.get("num_worker", 4),
                multiprocessing_context=self.args.get("mulc", None),
                persistent_workers=True,
            )
            self.val_loader.append(val_loader)

        self.test_loader = []
        for test_dataset in test_datasets:
            test_loader = DataLoader(
                test_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=self.args.get("num_worker", 4),
                multiprocessing_context=self.args.get("mulc", None),
                persistent_workers=True,
            )
            self.test_loader.append(test_loader)

        setup_seed(self.seed)
        if self._cur_task == 0 and (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
        if self._cur_task != 0:
            # get syn_data for old tasks
            self.syn_data_loader = self.get_syn_data_loader()

        # * for all tasks
        self._fl_train(train_dataset[0], self.val_loader, self.test_loader)

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

        if self._cur_task + 1 != self.tasks:
            self.data_generation(data_manager)

    def _local_update(self, model, train_data_loader, lr):
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args.get("weight_decay", 5e-4),
        )
        total_loss = 0
        for iter in range(self.args["local_ep"]):
            for batch_idx, (images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                # Multi-label loss.
                loss = F.binary_cross_entropy_with_logits(output, labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter == 0:
                    total_loss += loss.detach()
            print(
                "---task {}, ep {}/{} =>  Loss: {}".format(
                    self._cur_task, iter + 1, self.args["local_ep"], loss.item()
                )
            )
        return model.state_dict(), total_loss

    def _local_finetune(
        self, teacher, model, train_data_loader, task_id, client_id, lr
    ):
        model.train()
        teacher.eval()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=self.args.get("weight_decay", 5e-4),
        )
        syn_data_iter = None
        if hasattr(self, "syn_data_loader") and self.syn_data_loader is not None:
            syn_data_iter = DataIter(self.syn_data_loader)

        total_loss = 0
        for it in range(self.args["local_ep"]):
            iter_loader = enumerate(train_data_loader)
            total_local = 0.0
            total_syn = 0.0
            total_ce_loss = 0
            total_kd_loss = 0

            for batch_idx, (images, labels) in iter_loader:
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                # Multi-label loss for the current task.
                loss_ce = F.binary_cross_entropy_with_logits(output, labels.float())
                total_local += images.shape[0]

                if syn_data_iter is not None:
                    syn_input, syn_label = syn_data_iter.next()
                    syn_input = syn_input.cuda()
                    syn_label = syn_label.cuda().float()

                    s_out = model(syn_input.detach())["logits"]
                    with torch.no_grad():
                        t_out = teacher(syn_input.detach())["logits"]
                        total_syn += syn_input.shape[0]

                    # Multi-label knowledge distillation with binary_cross_entropy_with_logits.
                    loss_kd = _KD_loss(
                        s_out[:, : self._known_classes],
                        t_out.detach(),
                        2,
                    )
                    loss = loss_ce + kd * loss_kd
                else:
                    loss_kd = torch.tensor(0.0, device=images.device)
                    loss = loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if it == 0:
                    total_loss += loss.detach()
                    total_ce_loss += loss_ce.detach()
                    total_kd_loss += (
                        loss_kd.detach() if isinstance(loss_kd, torch.Tensor) else 0
                    )

            print(
                "---task {}, ep {}/{} =>  CE: {}, KD: {}, Total: {}".format(
                    self._cur_task,
                    it + 1,
                    self.args["local_ep"],
                    loss_ce.item(),
                    loss_kd.item() if isinstance(loss_kd, torch.Tensor) else 0,
                    loss.item(),
                )
            )

        return model.state_dict(), total_syn, total_local, total_loss

    def _fl_train(self, train_dataset, val_loader, test_loader):
        self._network.cuda()
        self.best_model = copy.deepcopy(self._network.state_dict())
        self.best_val_auc = 0

        user_groups, _ = partition_data(
            train_dataset.muti_labels,
            beta=self.args["beta"],
            n_parties=self.args["num_users"],
        )
        com_rounds = (
            self.args.get("com_rounds", [self.args.get("com_round", 50)])[
                self._cur_task
            ]
            if isinstance(self.args.get("com_rounds"), list)
            else self.args.get("com_round", 50)
        )

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
                f"Resuming from checkpoint: task {self._cur_task}, round {start_round}/{com_rounds}"
            )
            if getattr(self, "logger", None) is not None:
                self.logger.log(
                    f"Resume checkpoint: task {self._cur_task}, com_round {start_round}/{com_rounds}"
                )
            if hasattr(self, "best_model"):
                self.best_model = checkpoint.get("best_model", self.best_model)
            self.best_val_auc = checkpoint.get("best_val_auc", self.best_val_auc)

        if start_round >= com_rounds:
            self._network.load_state_dict(self.best_model)
            torch.cuda.empty_cache()
            return

        prog_bar = tqdm(range(start_round, com_rounds))

        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(
                range(self.args["num_users"]), m, replace=False
            )
            for idx in idxs_users:
                local_train_loader = DataLoader(
                    DatasetSplit(train_dataset, user_groups[idx]),
                    batch_size=self.args["local_bs"],
                    shuffle=True,
                    num_workers=self.args.get("num_worker", 4),
                    multiprocessing_context=self.args.get("mulc", None),
                    persistent_workers=True,
                )
                if self._cur_task == 0:
                    w, _ = self._local_update(
                        copy.deepcopy(self._network),
                        local_train_loader,
                        scheduler.get_last_lr()[0],
                    )
                else:
                    w, total_syn, total_local, _ = self._local_finetune(
                        self._old_network,
                        copy.deepcopy(self._network),
                        local_train_loader,
                        self._cur_task,
                        idx,
                        scheduler.get_last_lr()[0],
                    )
                    if com == 0 and self._cur_task == 1:
                        print(
                            "\t \t client {}, local dataset size:{},  syntheic data size:{}".format(
                                idx, total_local, total_syn
                            )
                        )

                local_weights.append(copy.deepcopy(w))
                del local_train_loader, w
                torch.cuda.empty_cache()

            scheduler.step()

            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            # Select the best model by validation macro F1.
            val_auc = self._compute_auc(copy.deepcopy(self._network), val_loader)
            print(
                "Task {}, Epoch {}/{} =>  Val_AUC {:.4f}".format(
                    self._cur_task, com + 1, com_rounds, val_auc
                )
            )
            if val_auc >= self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_model = copy.deepcopy(self._network.state_dict())

            self._save_checkpoint(com + 1, optimizer, scheduler)

        # Load the best model.
        self._network.load_state_dict(self.best_model)
        del self.best_model
        torch.cuda.empty_cache()


def _KD_loss(pred, soft, T):
    return F.binary_cross_entropy_with_logits(pred / T, torch.sigmoid(soft / T)) * (
        T**2
    )
