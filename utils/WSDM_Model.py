import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.condEmbedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb, c_cemb, s_cemb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb, c_cemb, s_cemb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class AEN(nn.Module):
    def __init__(self, in_ch, tdim, dropout, attn=True, pca_fcel=False, embedding_type=0):
        super().__init__()
        self.pca_fcel = pca_fcel
        self.embedding_type = embedding_type
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, 1, stride=1),

        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, in_ch),
        )
        self.cond_proj1 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, in_ch),
        )
        self.cond_proj2 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, in_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, 1, stride=1),
        )
        if attn:
            self.attn = AttnBlock(in_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb, cemb, s_cemb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        if self.pca_fcel:
            if self.embedding_type == 0:
                h += self.cond_proj1(cemb)[:, :, None, None]
                h += self.cond_proj1(s_cemb)[:, :, None, None]
            else:
                h1 = self.cond_proj2(cemb)
                h1 = torch.sum(h1, dim=0)
                h += h1[:, :, None, None]
        else:
            h += self.cond_proj1(cemb)[:, :, None, None]
        h = self.block2(h)
        h = self.attn(h)
        h = h + x
        return h


class cond_embed(nn.Module):
    def __init__(self, in_ch, tdim):
        super().__init__()
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, in_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, in_ch),
        )

    def forward(self, x, temb, cemb, s_cemb):
        x += self.temb_proj(temb)[:, :, None, None]
        x += self.cond_proj(cemb)[:, :, None, None]
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True, aen=True, pca_fcel=False, embedding_type=0):
        super().__init__()
        self.pca_fcel = pca_fcel
        self.embedding_type = embedding_type
        if aen:
            self.aen = AEN(in_ch, tdim, dropout, attn, pca_fcel, embedding_type)
        else:
            self.aen = cond_embed(in_ch, tdim)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb, cemb, s_cemb):
        h = self.aen(x, temb, cemb, s_cemb)
        # h = h + x
        h = self.shortcut(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, num_labels, num_shapes, pca_fcel, embedding_type, ch, ch_mult, attn, aen, num_res_blocks,
                 dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.pca_fcel = pca_fcel
        self.aen = aen
        self.embedding_type = embedding_type
        if self.pca_fcel:
            if self.embedding_type == 0:
                self.cond_embedding1 = ConditionalEmbedding(num_labels, ch, tdim)
                self.cond_embedding2 = ConditionalEmbedding(num_shapes, ch, tdim)
            else:
                self.cond_embedding = ConditionalEmbedding(num_shapes, ch, tdim)
        else:
            self.cond_embedding = ConditionalEmbedding(num_labels, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), aen=self.aen, pca_fcel=self.pca_fcel,
                    embedding_type=self.embedding_type))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True, aen=self.aen, pca_fcel=self.pca_fcel,
                     embedding_type=self.embedding_type),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False, aen=self.aen, pca_fcel=self.pca_fcel,
                     embedding_type=self.embedding_type),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), aen=self.aen, pca_fcel=self.pca_fcel,
                    embedding_type=self.embedding_type))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, cls_label, shape_label):
        # Timestep embedding
        temb = self.time_embedding(t)
        if self.pca_fcel:
            if self.embedding_type == 0:
                cemb = self.cond_embedding1(cls_label)
                s_cemb = self.cond_embedding2(shape_label)
            else:
                label_vector = torch.stack([cls_label, shape_label])
                cemb = self.cond_embedding(label_vector)
                s_cemb = None
        else:
            cemb = self.cond_embedding(cls_label)
            s_cemb = None
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb, s_cemb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, cemb, s_cemb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb, s_cemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size,))
    y = model(x, t)
    print(y.shape)
