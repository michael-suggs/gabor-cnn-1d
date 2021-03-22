import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.modules import Conv1d, Module


class GaborConv1d(Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, bias=True, padding_mode='zeros'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv_layer: Conv1d = Conv1d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode,
        )


