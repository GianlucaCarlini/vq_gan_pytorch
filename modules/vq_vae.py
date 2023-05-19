import torch
import torch.nn as nn
from .blocks import ResidualLayer, ResidualBlock


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        depths,
        channel_multipliers,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.channel_multipliers = channel_multipliers

        self.embed_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.down_blocks = nn.ModuleList()

        for i, depth in enumerate(self.depths):
            self.down_blocks.append(
                ResidualBlock(
                    in_channels=self.embed_dim * self.channel_multipliers[i],
                    out_channels=self.embed_dim * self.channel_multipliers[i + 1]
                    if i < len(self.depths) - 1
                    else self.embed_dim * self.channel_multipliers[i],
                    depth=depth,
                    downsample=True if i < len(self.depths) - 1 else False,
                )
            )

        self.final_norm = nn.GroupNorm(
            num_groups=1, num_channels=self.embed_dim * self.channel_multipliers[-1]
        )
        self.final_embedding = nn.Conv2d(
            in_channels=self.embed_dim * self.channel_multipliers[-1],
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.embed_conv(x)

        for down_block in self.down_blocks:
            x = down_block(x)

        x = self.final_norm(x)
        x = self.final_embedding(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        depths,
        channel_multipliers,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.channel_multipliers = channel_multipliers

        self.embed_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim * self.channel_multipliers[0],
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.up_blocks = nn.ModuleList()

        for i, depth in enumerate(self.depths):
            self.up_blocks.append(
                ResidualBlock(
                    in_channels=self.embed_dim * self.channel_multipliers[i],
                    out_channels=self.embed_dim * self.channel_multipliers[i + 1]
                    if i < len(self.depths) - 1
                    else self.embed_dim * self.channel_multipliers[i],
                    depth=depth,
                    downsample=False,
                )
            )
            if i < len(self.depths) - 1:
                self.up_blocks.append(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )

        self.final_norm = nn.GroupNorm(
            num_groups=1, num_channels=self.embed_dim * self.channel_multipliers[-1]
        )
        self.final_embedding = nn.Conv2d(
            in_channels=self.embed_dim * self.channel_multipliers[-1],
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.embed_conv(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.final_norm(x)
        x = self.final_embedding(x)

        return x
