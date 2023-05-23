import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_dim=None,
        num_layers=None,
        channel_multipliers=None,
        kernel_size=4,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels

        if embed_dim is not None:
            self.embed_dim = embed_dim
        else:
            self.embed_dim = 32

        if num_layers is not None:
            self.num_layers = num_layers
        else:
            self.num_layers = 3

        if channel_multipliers is not None:
            self.channel_multipliers = channel_multipliers
        else:
            self.channel_multipliers = [1, 2, 4]

        self.activation = nn.ReLU()

        self.kernel_size = kernel_size

        self.layers = nn.ModuleList()

        self.initial_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.kernel_size,
            stride=2,
            padding=1,
        )
        self.initial_norm = nn.GroupNorm(num_groups=1, num_channels=self.embed_dim)

        for i in range(1, self.num_layers, 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.embed_dim * self.channel_multipliers[i - 1],
                        out_channels=self.embed_dim * self.channel_multipliers[i],
                        kernel_size=self.kernel_size,
                        stride=2,
                        padding=1,
                    ),
                    nn.GroupNorm(
                        num_groups=1,
                        num_channels=self.embed_dim * self.channel_multipliers[i],
                    ),
                    nn.ReLU(),
                )
            )

        self.final_conv = nn.Conv2d(
            in_channels=self.embed_dim * channel_multipliers[-1],
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = self.activation(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_conv(x)

        return x
