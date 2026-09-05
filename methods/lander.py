import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from methods.target import warmup
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb
from torch.autograd import Variable
from torchvision import transforms
from kornia import augmentation
import os, math
import torch.nn.init as init
from PIL import Image
from methods.generator import NLGenerator, NLGenerator_IN
import shutil
from transformers import CLIPModel, CLIPProcessor

syn_round = 16
synthesis_batch_size = 64

bn_mmt = 0.9
tau = 2
T = 20.0
student_train_step = 50

warmup = 10
lr_g = 0.002

bn = 1.0
oh = 0.5
adv = 1.0
nz = 256
g_steps = 200
r = 0.015
ltc = 5.0


def get_norm_and_transform():
    data_normalize = dict(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        # transforms.Resize((256, 256)),
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
        transforms.RandomRotation(10),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(**dict(data_normalize)),
    ])
    normalizer = Normalizer(**dict(data_normalize))
    return train_transform, normalizer


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


# normalizer = Normalizer(**dict(data_normalize))


def _collect_all_images(nums, root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            if nums != None:
                files.sort()
                files = files[:nums]
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
        if isinstance(data, (list, tuple)):
            return data[0]
        return data


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, self.root)
        self.transform = transform
        labels_file = os.path.join(self.root, "labels.npy")
        if os.path.exists(labels_file):
            self.labels = np.load(labels_file)
            if len(self.labels) != len(self.images):
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
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        tag = "with labels" if self.labels is not None else "without labels"
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s\n\t%s' % (
            self.root, len(self), self.transform, tag)


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

    pack = np.zeros((C, H * row + padding * (row - 1), W * col + padding * (col - 1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H + padding)
        w = (idx % col) * (W + padding)
        pack[:, h:h + H, w:w + W] = img
    return pack


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir != '':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images(imgs, col=col).squeeze()
        mode = 'L' if imgs.ndim == 2 else 'RGB'
        if imgs.ndim == 2:
            image = Image.fromarray(imgs, mode=mode)
        else:
            image = Image.fromarray(imgs.transpose(1, 2, 0), mode=mode)
        if size is not None:
            if isinstance(size, (list, tuple)):
                image = image.resize(size)
            else:
                w, h = image.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w * scale), int(h * scale)
                image = image.resize([_w, _h])
        image.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            if img.shape[0] == 1:
                pil_img = Image.fromarray(img.squeeze(), mode='L')
            else:
                pil_img = Image.fromarray(img.transpose(1, 2, 0))
            pil_img.save(output_filename + '-%d.png' % (idx))


class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

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
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                        self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data)

    def remove(self):
        self.hook.remove()


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False)
        self._idx += 1

    def get_dataset(self, nums=None, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform, nums=nums)


def kldiv(logits, targets, T=1.0, reduction='batchmean', eps=1e-8):
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


def custom_cross_entropy(preds, target):
    return F.binary_cross_entropy_with_logits(preds, target)


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


class NAYER():
    def __init__(self, teacher, student, generator, num_classes, img_size, iterations=100, lr_g=0.1, label_emb=None,
                 synthesis_batch_size=128, adv=0.0, bn=1, oh=1, r=1e-1, ltc=0.2, save_dir='run/fast', transform=None,
                 normalizer=None, device="gpu:0", warmup=10, bn_mmt=0, args=None, target_img_size=None):
        super(NAYER, self).__init__()
        self.teacher = teacher
        self.student = student
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.args = args
        self.label_emb = label_emb
        self.ltc = ltc
        self.r = r
        self.device = device

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.normalizer = normalizer
        self.target_img_size = target_img_size

        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.generator = generator.cuda().train()
        self.ep = 0
        self.ep_start = warmup + 1  # need more time to the student train well.
        self.prev_z = None

        self.aug = transforms.Compose([
            augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
            augmentation.RandomRotation(degrees=10),
            # augmentation.RandomHorizontalFlip(),
        ])

        self.mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=True, device=self.device)
        self.std = torch.tensor([0.25, 0.25, 0.25], requires_grad=True, device=self.device)

        self.bn_mmt = bn_mmt
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

    def synthesize(self, _cur_task=0):
        self._cur_task = _cur_task
        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        best_oh = 1e6

        # if self.ep % 200 == 0:
        #     self.generator = self.generator.reinit()

        best_inputs = None
        self.generator.re_init_le()

        targets, ys = self.generate_ys(cr=0.0)
        ys = ys.to(self.device)

        optimizer = torch.optim.Adam([
            {'params': self.generator.parameters()},
            {'params': [self.mean], 'lr': 0.01},
            {'params': [self.std], 'lr': 0.01}
        ], lr=self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations):
            inputs = self.generator(targets=targets)
            if self.target_img_size is not None:
                t_c, t_h, t_w = self.target_img_size
                if inputs.shape[-2:] != (t_h, t_w):
                    inputs = F.interpolate(inputs, size=(t_h, t_w), mode='bilinear', align_corners=False)
            inputs_aug = self.aug(inputs)
            if inputs_aug.shape[1] == 1:
                inputs_norm = inputs_aug.repeat(1, 3, 1, 1)
            else:
                inputs_norm = inputs_aug
            inputs_norm = (inputs_norm - self.mean[None, :, None, None]) / (self.std[None, :, None, None])
            output_list = self.teacher(inputs_norm)
            t_out = output_list["logits"]
            feature = output_list["att"]

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = custom_cross_entropy(t_out, ys.detach())

            if self.adv > 0 and (self.ep - 1 > self.ep_start):
                s_out = self.student(inputs_norm)["logits"]
                s_pred = (torch.sigmoid(s_out) > 0.5).float()
                t_pred = (torch.sigmoid(t_out) > 0.5).float()
                mask = (s_pred == t_pred).all(dim=1).float()
                kd_loss = F.binary_cross_entropy_with_logits(
                    s_out,
                    torch.sigmoid(t_out.detach()),
                    reduction='none',
                )
                loss_adv = -(kd_loss.sum(1) * mask).mean()  # adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)

            target_f = torch.stack([self.label_emb[t].squeeze(0) for t in targets], dim=0)
            target_f = target_f.to(feature.device)
            loss_f = torch.relu(torch.nn.functional.mse_loss(feature, target_f.detach()) - self.r)

            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.ltc * loss_f

            if loss_oh.item() < best_oh:
                best_oh = loss_oh
            print("%s - bn %s - bn %s - oh %s - adv %s -fr %s - %s - %s" % (
                it,
                float((loss_bn * self.bn).data.cpu().detach().numpy()),
                float(loss_bn.data.cpu().detach().numpy()),
                float((loss_oh).data.cpu().detach().numpy()),
                float((loss_adv).data.cpu().detach().numpy()),
                float(loss_f.data.cpu().detach().numpy()),
                str(self.mean.detach().cpu()[0]),
                str(self.std.detach().cpu()[0])))

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        if warmup <= self.ep:
            self.data_pool.add(best_inputs)
            self.student_train(self.student, self.teacher)

    def student_train(self, student, teacher):
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data()
        data_iter = DataIter(loader)
        criterion = KLDiv(T=T)

        prog_bar = tqdm(range(student_train_step))
        for _, com in enumerate(prog_bar):
            images = data_iter.next()
            images = images.cuda()
            with torch.no_grad():
                t_out_list = teacher(images)
                t_out = t_out_list["logits"]
                t_f = t_out_list["att"]
            s_out_list = student(images.detach())
            s_out = s_out_list["logits"]
            s_f = s_out_list["att"]
            loss_s = criterion(s_out, t_out.detach())
            s_loss_f = torch.nn.functional.mse_loss(s_f, t_f.detach())

            loss = loss_s + self.ltc * s_loss_f

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def get_all_syn_data(self):
        syn_dataset = UnlabeledImageDataset(self.save_dir, transform=self.transform, nums=self.args['nums'])
        syn_num_workers = int(self.args["num_worker"])
        loader = torch.utils.data.DataLoader(
            syn_dataset,
            batch_size=self.args["local_bs"],
            shuffle=True,
            num_workers=syn_num_workers,
            persistent_workers=True,
        )
        return loader

    def index2label(self, indices):
        label = torch.zeros(self.num_classes).float()
        label[indices] = 1.0
        return label

    def generate_ys(self, cr=0.0):
        # Sample only observed label combinations to match the real multi-label distribution.

        observed_combos = list(self.label_emb.keys())
        batch_size = self.synthesis_batch_size
        num_classes = self.num_classes

        sampled_indices = torch.randint(len(observed_combos), (batch_size,))
        # Pad each observed combo to num_classes length with zeros if necessary
        target = [observed_combos[idx] for idx in sampled_indices]
        ys = torch.stack([self.index2label(list(t)) for t in target], dim=0)
        if cr > 0:
            ys = ys * (1 - cr) + cr / (num_classes - 1) * (1 - ys)
        return target, ys

    def get_student(self):
        return self.student

    def get_teacher(self):
        return self.teacher


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
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


class ClipModel:
    def __init__(self, device='cuda'):
        self.initialized = False
        self.clip_model = None
        self.clip_processor = None
        self.local_model_path = os.path.join(os.path.expanduser('~'), "HuggingFace_Model/openai/clip-vit-patch32")
        self.device = str(device)
        self.class_texts = [
            "no acute cardiopulmonary findings",
            "evidence of pneumonia",
            "consistent with COVID-19 pneumonia",
            "an enlarged cardiomediastinum",
            "cardiomegaly",
            "lung opacity",
            "a lung lesion",
            "pulmonary edema",
            "consolidation",
            "atelectasis",
            "pneumothorax",
            "pleural effusion",
            "other pleural abnormalities",
            "rib fracture",
            "pulmonary infiltrates",
            "effusion",
            "a nodule",
            "a mass",
            "pleural thickening",
            "emphysema",
            "pulmonary fibrosis",
            "hernia",
        ]

    def _init_if_needed(self):
        if not self.initialized:
            self.clip_model = CLIPModel.from_pretrained(self.local_model_path, local_files_only=True).to(self.device)
            self.clip_model.eval()
            self.clip_processor = CLIPProcessor.from_pretrained(self.local_model_path, local_files_only=True)
            self.initialized = True

    def multilabel_to_indices(self, labels):
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        result = []
        for i in range(labels.size(0)):
            nonzero_indices = torch.nonzero(labels[i], as_tuple=True)[0].tolist()
            result.append(nonzero_indices)
        return result

    def combine_texts(self, label_indices):
        self._init_if_needed()
        findings = [self.class_texts[i] for i in label_indices if i < len(self.class_texts)]
        if not findings:
            return "a chest X-ray with no acute cardiopulmonary findings"
        prefix = "a chest X-ray showing"
        if len(findings) == 1:
            return f"{prefix} {findings[0]}"
        return f"{prefix} {', '.join(findings[:-1])}, and {findings[-1]}"

    def caculate_embdding(self, label_indices):
        # label_indices = self.multilabel_to_indices(labels)
        text_descriptions = [self.combine_texts(indices) for indices in label_indices]
        inputs = self.clip_processor(text=text_descriptions, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_model.get_text_features(**inputs)
        return text_embeddings


class LANDER(BaseLearner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self._network = IncrementalNet(args, False, False)
        self.device = torch.device(f"cuda:{args['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
        self.clip_model = ClipModel(device=self.device)
        self.r = r
        self.ltc = ltc
        self.transform, self.normalizer = get_norm_and_transform()
        self.best_model = None
        self.best_val_f1 = 0
        self.syn_data_loader = None
        self.label_embedding_cache = dict()

    def calculate_LTE(self, train_dataset):
        # Collect all labels.
        all_labels = train_dataset.muti_labels
        if isinstance(all_labels, torch.Tensor):
            all_labels = all_labels.cpu().numpy()

        # Build unique label combinations keyed by active-label indices.

        key = self.clip_model.multilabel_to_indices(all_labels)
        # Ensure every label combination is a tuple.
        unique_label_tuples = set(tuple(l) for l in key)
        unique_label_tuples = list(unique_label_tuples)

        # Compute CLIP embeddings in batches.
        for label_indices in unique_label_tuples:
            emb = self.clip_model.caculate_embdding([list(label_indices)])  # (1, emb_dim)
            self.label_embedding_cache[label_indices] = emb.detach().cpu()

        # print(self.label_embedding_cache.keys())
        print(f"Precomputed {len(unique_label_tuples)} unique label embeddings.")

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        test_f1, test_auc = self._compute_accuracy(self._old_network, self.test_loader)
        info = f"After Task {self._cur_task}: F1 {test_f1}, AUC {test_auc}"
        print(info)
        self.logger.log(info)

    def remove_syn_imgs(self):
        folder_path = os.path.join(self.save_dir, "task_{}".format(self._cur_task - 1))
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (folder_path, e))

    def data_generation(self):
        img_size = 224
        generator = NLGenerator_IN(
            ngf=64,
            img_size=img_size,
            nc=1,
            nl=10,
            label_emb=self.label_embedding_cache,
            le_emb_size=nz,
            sbz=synthesis_batch_size,
        )
        target_img_size = (1, 244, 244)

        student = copy.deepcopy(self._network)
        student.apply(weight_init)
        tmp_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        synthesizer = NAYER(
            copy.deepcopy(self._network),
            student,
            generator,
            num_classes=self._total_classes,
            img_size=(1, img_size, img_size),
            save_dir=tmp_dir,
            transform=self.transform,
            normalizer=self.normalizer,
            synthesis_batch_size=synthesis_batch_size,
            iterations=g_steps,
            warmup=warmup,
            lr_g=lr_g,
            adv=adv,
            bn=bn,
            oh=oh,
            ltc=self.ltc,
            r=self.r,
            device=self.device,
            bn_mmt=bn_mmt,
            args=self.args,
            label_emb=self.label_embedding_cache,
            target_img_size=target_img_size,
        )

        for it in range(syn_round + warmup):
            synthesizer.synthesize(self._cur_task)  # generate synthetic data
            if it > warmup:
                ms = synthesizer.get_student()
                test_accs = self._compute_accuracy(ms, self.test_loader)
                mt = synthesizer.get_teacher()
                test_acct = self._compute_accuracy(mt, self.test_loader)
                print("Student Test Acc: %s - %s" % (test_accs, test_acct))
                if wandb.log == 1:
                    wandb.log({'Task_{}, Student Accuracy'.format(self._cur_task): test_accs})

        print("For task {}, data generation completed! ".format(self._cur_task))

    def get_syn_data_loader(self):
        syn_bs = self.args["syn_bs"] * self.args["local_bs"]
        task_idx = self._cur_task - 1
        data_dir = os.path.join(self.save_dir, f"task_{task_idx}")
        if not os.path.exists(data_dir):
            return None
        syn_dataset = UnlabeledImageDataset(data_dir, transform=self.transform, nums=self.nums)
        syn_num_workers = int(self.args["num_worker"])
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset,
            batch_size=syn_bs,
            shuffle=True,
            num_workers=syn_num_workers,
            multiprocessing_context=self.args["mulc"],
            persistent_workers=True,
        )
        return syn_data_loader

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._cur_dataset = self.dataset_names[self._cur_task]
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.transform, self.normalizer = get_norm_and_transform()

        self._network.update_fc(self._total_classes)
        message = f"Learning on {self._cur_dataset}: {self._known_classes}-{self._total_classes}"
        print(message)
        self.logger.log(message)

        train_dataset = data_manager.get_dataset(begin=self._cur_task, end=self._cur_task, mode="train")
        val_datasets = data_manager.get_dataset(begin=0, end=self._cur_task, mode="val")
        test_datasets = data_manager.get_dataset(begin=0, end=self._cur_task, mode="test")

        val_num_workers = int(self.args["num_worker"])
        test_num_workers = int(self.args["num_worker"])
        self.val_loader = [
            DataLoader(
                dataset,
                batch_size=256,
                shuffle=False,
                num_workers=val_num_workers,
                multiprocessing_context=self.args["mulc"],
                persistent_workers=True,
            )
            for dataset in val_datasets
        ]
        self.test_loader = [
            DataLoader(
                dataset,
                batch_size=256,
                shuffle=False,
                num_workers=test_num_workers,
                multiprocessing_context=self.args["mulc"],
                persistent_workers=True,
            )
            for dataset in test_datasets
        ]

        self.calculate_LTE(train_dataset[0])

        setup_seed(self.seed)
        if self._cur_task == 0 and (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
        if self._cur_task != 0:
            self.syn_data_loader = self.get_syn_data_loader()

        user_groups, _ = partition_data(train_dataset[0].muti_labels, beta=self.args["beta"],
                                        n_parties=self.args["num_users"])
        self._fl_train(user_groups, train_dataset[0], self.val_loader, self.test_loader)

        if self._cur_task != self.tasks:
            torch.save(copy.deepcopy(self._network.state_dict()), 'store/{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                       .format(self._cur_dataset, self.args["net"], self.args["num_users"],
                               self.args["beta"], self.args["method"], self._cur_task, self.ltc, self.args['exp_name']))
            self.data_generation()

    def _fl_train(self, user_groups, train_dataset, val_loader, test_loader):
        self._network.cuda()
        self.best_model = copy.deepcopy(self._network.state_dict())
        self.best_val_auc = 0
        com_rounds = self.args["com_rounds"][self._cur_task]
        optimizer = torch.optim.SGD(self._network.parameters(), lr=self.args['local_lr'], momentum=0.9,
                                    weight_decay=self.args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

        start_round = 0
        checkpoint = self._load_checkpoint(optimizer, scheduler)
        if checkpoint is not None:
            start_round = int(checkpoint.get("com_round", 0))
            print(f"Resuming from checkpoint: task {self._cur_task}, round {start_round}/{com_rounds}")
            if self.logger is not None:
                self.logger.log(f"Resume checkpoint: task {self._cur_task}, com_round {start_round}/{com_rounds}")
            if hasattr(self, 'best_model'):
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
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            loss_weight = []
            current_lr = scheduler.get_last_lr()[0]
            for idx in idxs_users:
                local_train_loader = DataLoader(
                    DatasetSplit(train_dataset, user_groups[idx]),
                    batch_size=self.args["local_bs"],
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    # multiprocessing_context=self.args["mulc"],
                    persistent_workers=False,
                )
                if self._cur_task == 0:
                    w, total_loss = self._local_update(copy.deepcopy(self._network), local_train_loader, current_lr)
                else:
                    w, total_syn, total_local, total_loss = self._local_finetune(
                        self._old_network, copy.deepcopy(self._network), local_train_loader, self._cur_task, idx,
                        current_lr)
                    if com == 0 and self._cur_task == 1:
                        print(f"\t \t client {idx}, local dataset size:{total_local}, synthetic data size:{total_syn}")
                local_weights.append(copy.deepcopy(w))
                loss_weight.append(total_loss)
                del local_train_loader, w
                torch.cuda.empty_cache()
            scheduler.step()

            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            val_auc = self._compute_auc(copy.deepcopy(self._network), val_loader)
            print("--- Task {} Round {} Validation AUC: {}".format(self._cur_task, com + 1, val_auc))
            if val_auc >= self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_model = copy.deepcopy(self._network.state_dict())

            self._save_checkpoint(com + 1, optimizer, scheduler)

        self._network.load_state_dict(self.best_model)
        del self.best_model
        torch.cuda.empty_cache()

    def _local_update(self, model, train_data_loader, lr):
        model.train()
        total_loss = 0
        total_ce_loss = 0
        total_f_loss = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args['weight_decay'])
        for iter in range(self.args["local_ep"]):
            for batch_idx, (images, labels) in enumerate(train_data_loader):
                images = images.cuda()
                labels = labels.cuda().float()
                output_list = model(images)
                output = output_list["logits"]
                feature = output_list["att"]
                lable_indices = self.clip_model.multilabel_to_indices(labels.cpu())
                label_tuples = [tuple(l) for l in lable_indices]
                target_f = torch.stack([self.label_embedding_cache[t].squeeze(0) for t in label_tuples], dim=0)
                target_f = target_f.to(feature.device)

                loss_f = torch.relu(torch.nn.functional.mse_loss(feature, target_f.detach()) - self.r)

                loss_ce = F.binary_cross_entropy_with_logits(output, labels)
                loss = loss_ce + self.ltc * loss_f
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter == 0:
                    total_loss += loss.detach()
                    total_ce_loss += loss_ce.detach()
                    total_f_loss += loss_f.detach()
        print("---task {} =>  CE: {} F: {}, CE: {}, F: {} T: {}".format(
            self._cur_task, loss_ce.detach(), loss_f.detach(), total_ce_loss, total_f_loss, total_loss))

        return model.state_dict(), total_loss

    def _local_finetune(self, teacher, model, train_data_loader, task_id, client_id, lr):
        pre = self.args['pre']
        cur = self.args['cur']

        alpha = np.log2(self._total_classes / 2 + 1)
        beta = np.sqrt(self._known_classes / self._total_classes)
        cur = cur * (1 + 1 / alpha) / beta
        pre = pre * alpha * beta

        # global print_flag
        model.train()
        teacher.eval()
        total_loss = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args['weight_decay'])
        syn_data_iter = DataIter(self.syn_data_loader) if self.syn_data_loader is not None else None
        for it in range(self.args["local_ep"]):
            iter_loader = enumerate(train_data_loader)
            total_local = 0.0
            total_syn = 0.0
            total_ce_loss = 0
            total_f_loss = 0
            total_kd_loss = 0
            total_sf_loss = 0

            for batch_idx, (images, labels) in iter_loader:
                images = images.cuda()
                labels = labels.cuda().float()

                syn_input = None
                if syn_data_iter is not None:
                    syn_input = syn_data_iter.next()
                    syn_input = syn_input.cuda()
                c_output_list = model(images)
                c_output = c_output_list["logits"]
                c_feature = c_output_list["att"]

                lable_indices = self.clip_model.multilabel_to_indices(labels.cpu())
                label_tuples = [tuple(l) for l in lable_indices]
                target_f = torch.stack([self.label_embedding_cache[t].squeeze(0) for t in label_tuples], dim=0)

                c_target_f = target_f.to(c_feature.device)
                c_loss_f = torch.relu(torch.nn.functional.mse_loss(c_feature, c_target_f.detach()) - self.r)

                loss_ce_cur = F.binary_cross_entropy_with_logits(c_output, labels)

                loss_kd = torch.tensor(0.0, device=images.device)
                s_loss_f = torch.tensor(0.0, device=images.device)
                s_out_list = model(syn_input.detach())
                s_out = s_out_list["logits"]
                s_f = s_out_list["att"]

                with torch.no_grad():
                    t_out_list = teacher(syn_input.detach())
                    t_out = t_out_list["logits"]
                    t_f = t_out_list["att"]
                    total_syn += syn_input.shape[0]
                    total_local += images.shape[0]
                loss_kd = _KD_loss(
                    s_out[:, : self._known_classes],
                    t_out.detach(),
                    tau,
                )
                s_loss_f = torch.nn.functional.mse_loss(s_f, t_f.detach())

                loss = cur * (loss_ce_cur + self.ltc * c_loss_f) + pre * (loss_kd + s_loss_f)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if it == 0:
                    total_loss += loss.detach()
                    total_ce_loss += loss_ce_cur.detach()
                    total_f_loss += c_loss_f.detach()
                    total_kd_loss += loss_kd.detach()
                    total_sf_loss += s_loss_f.detach()
            print("---task {}, ep {}/{} =>  CE: {} F: {} CE: {}, F: {}. TKL: {}, TF: {}, T: {}".format(
                self._cur_task, iter, self.args["local_ep"], loss_ce_cur.detach(), c_loss_f.detach(), total_ce_loss,
                total_f_loss, total_kd_loss, total_sf_loss, total_loss))

        return model.state_dict(), total_syn, total_local, total_loss


def _KD_loss(pred, soft, T):
    return F.binary_cross_entropy_with_logits(pred / T, torch.sigmoid(soft / T)) * (T ** 2)
