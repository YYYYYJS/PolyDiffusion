from typing import Tuple

import torch
import torch.nn as nn


# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


# 旋转位置编码计算
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SelfAttentionModule(nn.Module):
    def __init__(self, width, num_heads):
        super(SelfAttentionModule, self).__init__()

        # Define linear transformations for image and text
        self.q_linear = nn.Linear(width, width)
        self.k_linear = nn.Linear(width, width)
        self.v_linear = nn.Linear(width, width)

        self.freqs_cis = precompute_freqs_cis(width, 97).to('cuda')
        self.embed_dim = width
        # Self-attention layers for image and text
        self.compute_attention = nn.MultiheadAttention(embed_dim=width, num_heads=num_heads)
        self.out = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, tensor, mask=None):
        # Apply linear transformations to image and text tensors
        q_proj = self.q_linear(tensor)
        k_proj = self.k_linear(tensor)
        v_proj = self.v_linear(tensor)
        q_proj = q_proj.permute(1, 0, 2)
        k_proj = k_proj.permute(1, 0, 2)
        q_proj, k_proj = apply_rotary_emb(q_proj, k_proj, freqs_cis=self.freqs_cis)
        q_proj = q_proj.permute(1, 0, 2)
        k_proj = k_proj.permute(1, 0, 2)
        # Calculate cross-attention between image and text
        attention_output, _ = self.compute_attention(query=q_proj, key=k_proj, value=v_proj, attn_mask=mask)
        attention_output = self.out(attention_output)
        return attention_output


class CrossAttentionModule(nn.Module):
    def __init__(self, q_channels, k_channels, num_heads):
        super(CrossAttentionModule, self).__init__()

        # Define linear transformations for image and text
        self.q_linear = nn.Linear(q_channels, q_channels)
        self.k_linear = nn.Linear(k_channels, q_channels)
        self.embed_dim = q_channels
        # Self-attention layers for image and text
        self.compute_attention = nn.MultiheadAttention(embed_dim=q_channels, num_heads=num_heads)
        self.out = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, q_tensor, k_tensor):
        # Apply linear transformations to image and text tensors
        q_proj = self.q_linear(q_tensor)
        k_proj = self.k_linear(k_tensor)

        # Calculate cross-attention between image and text
        attention_output, _ = self.compute_attention(query=q_proj, key=k_proj, value=k_proj)
        attention_output = self.out(attention_output)
        return attention_output


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, num_classes, class_size, embedding_size, num_heads, layer, length):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Linear(input_size, embedding_size)
        self.position_embedding = nn.Parameter(torch.empty(length, embedding_size))
        self.atten_layers = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=class_size)
        for i in range(layer):
            self.atten_layers.append(SelfAttentionModule(embedding_size, num_heads))
            self.atten_layers.append(CrossAttentionModule(embedding_size, class_size, num_heads))

    def forward(self, inputs, class_label, mask):
        embedded = self.embedding(inputs) + self.position_embedding[None]

        class_embedding = self.class_embedding(class_label).unsqueeze(1)
        class_embedding = class_embedding.permute(1, 0, 2)
        embedded = embedded.permute(1, 0, 2)
        if mask is not None:
            mask = mask.repeat(8, 1, 1)
        for i in range(len(self.atten_layers)):
            if i % 2 == 0:
                embedded = self.atten_layers[i](self.layer_norm(embedded), mask) + embedded

            else:
                embedded = self.atten_layers[i](self.layer_norm(embedded), class_embedding) + embedded

        embedded = embedded.permute(1, 0, 2)
        return embedded


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, class_size, embedding_size, num_heads, layer):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(input_size, num_classes, class_size, embedding_size, num_heads, layer, 97)

        self.class_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=class_size)
        self.decoder = nn.Sequential(nn.Linear(embedding_size, embedding_size * 2), nn.SiLU(),
                                     nn.Linear(embedding_size * 2, input_size
                                               ))  # Output box coordinates
        self.box_regressor = nn.Sequential(nn.Linear(embedding_size, embedding_size * 2),
                                           nn.SiLU(),
                                           nn.Linear(embedding_size * 2, 4
                                                     ))  # Output box coordinates

        self.class_proj = nn.Sequential(nn.Linear(embedding_size, 2 * embedding_size), nn.SiLU(),
                                        nn.Linear(2 * embedding_size, num_classes))

    def forward(self, inputs, class_label, mask=None):
        cross_output = self.encoder(inputs, class_label, mask)

        box_coordinates = self.box_regressor(cross_output[:, 0, :])

        class_pre = self.class_proj(cross_output[:, 0, :])

        reconstructed_coords = self.decoder(cross_output[:, 1:, :])
        box_coordinates = box_coordinates.view(inputs.shape[0], 2, 2)
        return reconstructed_coords, box_coordinates, class_pre
