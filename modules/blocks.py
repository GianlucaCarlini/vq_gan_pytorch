import torch
import torch.nn as nn
from functools import partial


class ResidualLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation=None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if activation is not None:
            self.activation = activation
        else:
            self.activation = nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.gn1 = nn.GroupNorm(num_groups=1, num_channels=self.in_channels)

        self.conv2 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.gn2 = nn.GroupNorm(num_groups=1, num_channels=self.in_channels)

        if self.in_channels != self.out_channels:
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.projection_residual = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.projection = None
            self.projection_residual = None

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.activation(x)

        if self.projection is not None:
            residual = self.projection_residual(residual)

        if self.projection is not None:
            x = self.projection(x)

        x += residual
        x = self.activation(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, depth, downsample=False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.activation = kwargs.get("activation", nn.ReLU())

        if downsample:
            self.downsample = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            )
        else:
            self.downsample = None

        self.residual_blocks = nn.ModuleList()

        for i in range(depth):
            if i == depth - 1:
                self.residual_blocks.append(
                    ResidualLayer(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        activation=self.activation,
                    )
                )

            else:
                self.residual_blocks.append(
                    ResidualLayer(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        activation=self.activation,
                    )
                )

    def forward(self, x):
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
