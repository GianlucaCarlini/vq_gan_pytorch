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


class VectorQuantizationLayer(nn.Module):
    def __init__(
        self, num_vectors=None, vector_dimension=None, beta=None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        if num_vectors is not None:
            self.num_vectors = num_vectors
        else:
            self.num_vectors = 1024

        if vector_dimension is not None:
            self.vector_dimension = vector_dimension
        else:
            self.vector_dimension = 4

        if beta is not None:
            self.beta = beta
        else:
            self.beta = 0.25

        self.embedding = nn.Embedding(
            num_embeddings=self.num_vectors, embedding_dim=self.vector_dimension
        )

        self.embedding.weight.data.uniform_(-1 / self.num_vectors, 1 / self.num_vectors)

    def get_indices(self, flatten_inputs):
        # flatten inputs is NxC dimensional, embedding is KxC dimensional
        # so we have to transpose the embedding matrix
        # similarity will be an NxK dimensional matrix
        similarity = torch.matmul(flatten_inputs, self.embedding.weight.t())

        # here we calculate the L2 distance
        distances = (
            torch.sum(flatten_inputs**2, axis=1, keepdims=True)
            + torch.sum(self.embedding.weight**2, axis=1)
            - 2 * similarity
        )

        encoding_indices = torch.argmin(distances, axis=1).unsqueeze(1)

        return encoding_indices

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        flatten = x.view(-1, self.vector_dimension)

        encoding_indices = self.get_indices(flatten)

        # that's like a trick to get the one-hot encoding
        # instead of using torch.nn.functional.one_hot
        # we use torch.zeros and scatter the ones to the indices positions
        # I think this is done because torch one_hot takes long tensors
        encodings = torch.zeros(size=(encoding_indices.shape[0], self.num_vectors)).to(
            x
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)

        commitment_loss = torch.mean((quantized.detach() - x) ** 2)
        codebook_loss = torch.mean((quantized - x.detach()) ** 2)

        loss = codebook_loss + self.beta * commitment_loss

        # during the forward pass the two input terms cancel out (x and - x)
        # at backprop time, the stop gradient will exclude (quantized - x)
        # from the graph, so the gradient of quantized is actually copied to inputs
        # this is the implementation of the straight-through pass of VQ-VAE paper
        quantized = x + (quantized - x).detach()

        # perpexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, loss, perplexity
