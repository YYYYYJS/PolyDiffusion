from functools import partial
from typing import Callable
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            d_state: int = 64,
            **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim).type(torch.float32)
        self.mamba = Mamba(d_model=hidden_dim, d_state=d_state, **kwargs)

    def forward(self, input: torch.Tensor):
        x = input + self.mamba(self.norm(input))
        return x


class MambaEncoder(nn.Module):
    def __init__(self, input_size, num_classes, class_size, embedding_size, depth):
        super(MambaEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, embedding_size).type(torch.float16)

        self.layers = nn.ModuleList()
        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=class_size).type(torch.float16)
        for i in range(depth):
            self.layers.append(VSSBlock(hidden_dim=embedding_size, d_state=64))

    def forward(self, inputs, class_label):
        embedded = self.embedding(inputs)
        class_embedding = self.class_embedding(class_label).unsqueeze(1)
        embedded = embedded + class_embedding
        # embedded = embedded.to(torch.float32)
        for i in range(len(self.layers)):
            embedded = self.layers[i](embedded)

        embedded = embedded.to(torch.float16)
        return embedded
