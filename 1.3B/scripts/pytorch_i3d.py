"""PyTorch Inception-I3D network used by the standard Kinetics-400 FVD.

The module names intentionally match the public RGB weights released with
https://github.com/piergiaj/pytorch-i3d (Carreira & Zisserman, CVPR 2017).
Only inference code required for FVD is included here.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaxPool3dSamePadding(nn.MaxPool3d):
    def _padding(self, size: int, dim: int) -> tuple[int, int]:
        stride = self.stride[dim]
        kernel = self.kernel_size[dim]
        total = max(kernel - stride if size % stride == 0 else kernel - size % stride, 0)
        return total // 2, total - total // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pt = self._padding(x.shape[2], 0)
        ph = self._padding(x.shape[3], 1)
        pw = self._padding(x.shape[4], 2)
        return super().forward(F.pad(x, (pw[0], pw[1], ph[0], ph[1], pt[0], pt[1])))


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        kernel_shape: tuple[int, int, int] | list[int] = (1, 1, 1),
        stride: tuple[int, int, int] = (1, 1, 1),
        activation_fn=F.relu,
        use_batch_norm: bool = True,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_shape = tuple(kernel_shape)
        self.stride_shape = tuple(stride)
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.conv3d = nn.Conv3d(
            in_channels,
            output_channels,
            kernel_size=self.kernel_shape,
            stride=self.stride_shape,
            padding=0,
            bias=use_bias,
        )
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.01)

    def _padding(self, size: int, dim: int) -> tuple[int, int]:
        stride = self.stride_shape[dim]
        kernel = self.kernel_shape[dim]
        total = max(kernel - stride if size % stride == 0 else kernel - size % stride, 0)
        return total // 2, total - total // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pt = self._padding(x.shape[2], 0)
        ph = self._padding(x.shape[3], 1)
        pw = self._padding(x.shape[4], 2)
        x = self.conv3d(F.pad(x, (pw[0], pw[1], ph[0], ph[1], pt[0], pt[1])))
        if self.use_batch_norm:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: list[int]) -> None:
        super().__init__()
        self.b0 = Unit3D(in_channels, out_channels[0])
        self.b1a = Unit3D(in_channels, out_channels[1])
        self.b1b = Unit3D(out_channels[1], out_channels[2], (3, 3, 3))
        self.b2a = Unit3D(in_channels, out_channels[3])
        self.b2b = Unit3D(out_channels[3], out_channels[4], (3, 3, 3))
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.b3b = Unit3D(in_channels, out_channels[5])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (self.b0(x), self.b1b(self.b1a(x)), self.b2b(self.b2a(x)), self.b3b(self.b3a(x))),
            dim=1,
        )


class InceptionI3d(nn.Module):
    """Inception-v1 I3D with module names compatible with the public weights."""

    def __init__(self, num_classes: int = 400, dropout_keep_prob: float = 0.5) -> None:
        super().__init__()
        endpoints: list[tuple[str, nn.Module]] = [
            ("Conv3d_1a_7x7", Unit3D(3, 64, (7, 7, 7), (2, 2, 2))),
            ("MaxPool3d_2a_3x3", MaxPool3dSamePadding((1, 3, 3), stride=(1, 2, 2))),
            ("Conv3d_2b_1x1", Unit3D(64, 64)),
            ("Conv3d_2c_3x3", Unit3D(64, 192, (3, 3, 3))),
            ("MaxPool3d_3a_3x3", MaxPool3dSamePadding((1, 3, 3), stride=(1, 2, 2))),
            ("Mixed_3b", InceptionModule(192, [64, 96, 128, 16, 32, 32])),
            ("Mixed_3c", InceptionModule(256, [128, 128, 192, 32, 96, 64])),
            ("MaxPool3d_4a_3x3", MaxPool3dSamePadding((3, 3, 3), stride=(2, 2, 2))),
            ("Mixed_4b", InceptionModule(480, [192, 96, 208, 16, 48, 64])),
            ("Mixed_4c", InceptionModule(512, [160, 112, 224, 24, 64, 64])),
            ("Mixed_4d", InceptionModule(512, [128, 128, 256, 24, 64, 64])),
            ("Mixed_4e", InceptionModule(512, [112, 144, 288, 32, 64, 64])),
            ("Mixed_4f", InceptionModule(528, [256, 160, 320, 32, 128, 128])),
            ("MaxPool3d_5a_2x2", MaxPool3dSamePadding((2, 2, 2), stride=(2, 2, 2))),
            ("Mixed_5b", InceptionModule(832, [256, 160, 320, 32, 128, 128])),
            ("Mixed_5c", InceptionModule(832, [384, 192, 384, 48, 128, 128])),
        ]
        self.endpoint_names: list[str] = []
        for name, module in endpoints:
            self.add_module(name, module)
            self.endpoint_names.append(name)
        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            1024,
            num_classes,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in self.endpoint_names:
            x = self._modules[name](x)
        x = self.logits(self.dropout(self.avg_pool(x)))
        # With 16x224x224 input this is [N, 400, 1, 1, 1].  Averaging the
        # remaining dimensions also makes the behavior explicit and robust.
        return x.mean(dim=(2, 3, 4))
