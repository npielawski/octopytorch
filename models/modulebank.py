from enum import Enum, auto
from functools import partial
from collections.abc import Callable
from typing import Dict
from torch import nn


class ModuleType(Enum):
    """Module names that define which layers to use in the Tiramisu model."""

    CONV = auto()  # Convolution operations
    CONV_INIT = auto()  # Initial (1st) conv. operation. Kernel size must be provided.
    CONV_FINAL = auto()  # Final convolution. 1x1 kernel and reduce output to C classes.
    BATCHNORM = auto()  # Batch normalization
    POOLING = auto()  # Pooling operation (must reduce input size by a factor of two)
    # Note: if the size is odd, round *up* to the closest integer.
    DROPOUT = auto()  # Dropout
    UPSAMPLE = auto()  # Upsampling operation (must be by a factor of two)
    ACTIVATION = auto()  # Activation function to use everywhere
    ACTIVATION_FINAL = auto()  # Act. function at the last layer (e.g.softmax)


ModuleBankType = Dict[ModuleType, Callable[..., nn.Module]]
"""Type of module banks"""

UPSAMPLE2D_NEAREST = lambda in_channels, out_channels, module_bank: nn.Sequential(
    nn.UpsamplingNearest2d(scale_factor=2),
    module_bank[ModuleType.CONV](
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
    ),
    module_bank[ModuleType.ACTIVATION](),
)
# Pixel shuffle see: https://arxiv.org/abs/1609.05158
UPSAMPLE2D_PIXELSHUFFLE = lambda in_channels, out_channels, module_bank: nn.Sequential(
    module_bank[ModuleType.CONV](
        in_channels=in_channels,
        out_channels=4 * out_channels,
        kernel_size=3,
        padding=1,
    ),
    module_bank[ModuleType.ACTIVATION](),
    nn.PixelShuffle(2),
)
UPSAMPLE2D_TRANPOSE = lambda in_channels, out_channels, module_bank: nn.Sequential(
    nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=2,
        padding=0,
    ),
    module_bank[ModuleType.ACTIVATION](),
)
DEFAULT_MODULE_BANK: ModuleBankType = {
    ModuleType.CONV: nn.Conv2d,
    ModuleType.CONV_INIT: partial(nn.Conv2d, kernel_size=3, padding=1),
    ModuleType.CONV_FINAL: nn.Conv2d,
    ModuleType.BATCHNORM: nn.BatchNorm2d,
    ModuleType.POOLING: partial(nn.MaxPool2d, kernel_size=2, ceil_mode=True),
    ModuleType.DROPOUT: partial(nn.Dropout2d, p=0.2, inplace=True),
    ModuleType.UPSAMPLE: UPSAMPLE2D_NEAREST,
    ModuleType.ACTIVATION: partial(nn.ReLU, inplace=True),
    ModuleType.ACTIVATION_FINAL: nn.Identity,
}
