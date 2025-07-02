from functools import partial
from typing import Callable
import torch
import torch.nn as nn
from timm.layers import DropPath


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 128,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 64,
            **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.mamba(self.norm(input)))
        return x


class MambaEncoder(nn.Module):
    def __init__(self, input_size, num_classes, class_size, embedding_size, depth, length):
        super(MambaEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, embedding_size)
        self.position_embedding = nn.Parameter(torch.empty(length, embedding_size))
        self.layers = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=class_size)
        for i in range(depth):
            self.layers.append(VSSBlock(hidden_dim=embedding_size, d_state=64))

    def forward(self, inputs, class_label):
        embedded = self.embedding(inputs) + self.position_embedding[None]
        class_embedding = self.class_embedding(class_label).unsqueeze(1)
        embedded = embedded + class_embedding

        for i in range(len(self.layers)):
            embedded = self.layers[i](embedded)

        return embedded[:, -1, :]
