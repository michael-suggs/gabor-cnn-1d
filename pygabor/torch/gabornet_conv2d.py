"""Implementation of GaborConv2d torch module from [1]_.

Citations
---------
.. [1] https://github.com/iKintosh/GaborNet

"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu
from torch.nn.modules import Conv2d, Module


class GaborConv2d(Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, bias=False, padding_mode='zeros',
    ) -> None:
        super().__init__()
        self.is_calculated: bool = False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: Union[int, Tuple[int, int]] = kernel_size
        self.delta: float = 1e-3

        self.conv_layer: Conv2d = Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode,
        )

        self.freq = nn.Parameter(
            (math.pi / 2) * math.sqrt(2)
            ** (-torch.randint(0, 5, (self.out_channels, self.in_channels)))
            .type(torch.Tensor),
            requires_grad=True,
        )

        self.theta = nn.Parameter(
            (math.pi / 8) * torch.randint(0, 8,
                                          (self.out_channels, self.in_channels)).type(torch.Tensor),
            requires_grad=True,
        )

        self.sigma = nn.Parameter(math.pi / self.freq,
                                  requires_grad=True)

        self.psi = nn.Parameter(
            math.pi * torch.rand(self.out_channels, self.in_channels),
            requires_grad=True,
        )

        self.x0 = nn.Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0],
            requires_grad=True,
        )

        self.y0 = nn.Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0],
            requires_grad=True,
        )

        self.y, self.x = nn.Parameter(y), nn.Parameter(x) = (
            torch.meshgrid([
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
            ])
        )

        self.weight = nn.Parameter(
            torch.empty(self.conv_layer.weight.shape, requires_grad=True),
            requires_grad=True,
        )

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor) -> Conv2d:
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training and not self.is_calculated:
            self.calculate_weights()
            self.is_calculated = True
        return self.conv_layer(input_tensor)

    def calculate_weights(self):
        for i in range(self.conv_layer.out_channels):
            for j in range(self.conv_layer.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                self.conv_layer.weight.data[i, j] = (
                    torch.exp(
                        -0.5 * ((rotx ** 2 + roty ** 2)
                                / (sigma + self.delta) ** 2)
                    )
                    * torch.cos(freq * rotx + psi)
                    / (2 * math.pi * sigma ** 2)
                )


class GaborCNN2D(Module):
    def __init__(self, layers: Optional[List[Module]] = None):
        super(GaborCNN2D, self).__init__()
        if not layers:
            self.layers = [
                GaborConv2d(in_channels=1, out_channels=96,
                            kernel_size=(11, 11)),
                nn.Conv2d(96, 384, (3, 3)),
                nn.Linear(384*3*3, 64),
                nn.Linear(64, 2),
            ]
        else:
            self.layers = layers

    def forward(self, x):
        x = leaky_relu(self.layers[0](x))
        x = nn.MaxPool2d()(x)
        x = leaky_relu(self.layers[1](x))
        x = nn.MaxPool2d()(x)
        x = x.view(-1, 384*3*3)
        x = leaky_relu(self.layers[2](x))
        x = self.layers[3](x)
        return x
