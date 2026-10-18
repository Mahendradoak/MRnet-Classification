"""Slice encoder + multiple-instance aggregation + plane fusion.

An exam of one plane is a bag of S slices x_1..x_S. Each slice is encoded by a
2D CNN, h_s = f(x_s), and the bag embedding is either

* max pooling (the MRNet baseline):     z = max_s h_s
* gated attention MIL (Ilse et al., ICML 2018):
      a_s = softmax_s( w^T [tanh(V h_s) * sigmoid(U h_s)] ),   z = sum_s a_s h_s

The attention weights a_s say which slices drove the prediction. They replace
the original "spatial attention", whose softmax-then-divide-by-max normalisation
was not a distribution over anything.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def slice_encoder(pretrained: bool = True) -> nn.Module:
    """ResNet18 trunk for 1-channel input. The pretrained RGB conv1 kernel is
    summed over its input channels instead of being thrown away."""
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    w = net.conv1.weight.detach().sum(dim=1, keepdim=True)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        net.conv1.weight.copy_(w)
    net.fc = nn.Identity()
    return net


class GatedAttentionPool(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 128):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)   # (S,)
        a = F.softmax(a, dim=0)
        return (a[:, None] * h).sum(0), a


class PlaneNet(nn.Module):
    """One plane -> per-task logits."""

    def __init__(self, n_tasks: int, pool: str = "attention", pretrained: bool = True, chunk: int = 32):
        super().__init__()
        self.encoder, self.pool_kind, self.chunk = slice_encoder(pretrained), pool, chunk
        self.pool = GatedAttentionPool() if pool == "attention" else None
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n_tasks))

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = torch.cat([self.encoder(x[i:i + self.chunk]) for i in range(0, x.shape[0], self.chunk)])
        if self.pool is None:
            return h.max(0).values, None
        return self.pool(h)

    def forward(self, x: torch.Tensor) -> dict:
        z, a = self.embed(x)
        return {"logits": self.head(z), "attention": a}


class MultiPlaneNet(nn.Module):
    """Three plane-specific encoders; embeddings of the SAME exam are concatenated."""

    def __init__(self, n_tasks: int, planes: tuple[str, ...], pool: str = "attention", pretrained: bool = True):
        super().__init__()
        self.planes = planes
        self.branches = nn.ModuleDict({p: PlaneNet(n_tasks, pool, pretrained) for p in planes})
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512 * len(planes), n_tasks))

    def forward(self, exam: dict) -> dict:
        zs, att = [], {}
        for p in self.planes:
            z, a = self.branches[p].embed(exam[p])
            zs.append(z)
            att[p] = a
        return {"logits": self.head(torch.cat(zs)), "attention": att}
