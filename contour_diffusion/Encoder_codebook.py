from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F




def bilinear_interpolate(D, x, y):
    """
    对特征图D在(x, y)坐标处进行双线性插值

    参数:
        D: 特征图张量,形状为(height, width, channels)
        x: 要插值的x坐标,浮点数
        y: 要插值的y坐标,浮点数

    返回:
        插值后的向量,形状为(channels,)
    """
    H = D.shape[0]
    D = D.unsqueeze(0)  # (1, height, width, channels)
    # 计算周围四个整数坐标
    x0 = torch.floor(x * H).long()
    x1 = x0 + 1
    y0 = torch.floor(y * H).long()
    y1 = y0 + 1

    # 裁剪坐标,以免越界
    x0 = torch.clamp(x0, 0, D.shape[2] - 1)
    x1 = torch.clamp(x1, 0, D.shape[2] - 1)
    y0 = torch.clamp(y0, 0, D.shape[1] - 1)
    y1 = torch.clamp(y1, 0, D.shape[1] - 1)

    # 获取四个顶点的嵌入向量
    v00 = D[:, y0, x0]
    v01 = D[:, y0, x1]
    v10 = D[:, y1, x0]
    v11 = D[:, y1, x1]

    # 计算插值权重
    wx0 = x1 - (x * H)
    wx1 = (x * H) - x0
    wy0 = y1 - (y * H)
    wy1 = (y * H) - y0

    # 执行双线性插值
    interpolated = wx0 * wy0 * v00 + wx1 * wy0 * v01 + wx0 * wy1 * v10 + wx1 * wy1 * v11

    return interpolated.squeeze(0)  # 移除批量维度


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

    def forward(self, tensor, mask):
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

        self.codebook = torch.randn((1, 25, 25, 128), dtype=torch.float32).repeat(8, 1, 1, 1).to('cuda')

        self.position_embedding = nn.Parameter(torch.empty(length, embedding_size))
        self.atten_layers = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=class_size)
        for i in range(layer):
            self.atten_layers.append(SelfAttentionModule(embedding_size, num_heads))
            self.atten_layers.append(CrossAttentionModule(embedding_size, class_size, num_heads))

    def forward(self, inputs, class_label, mask):

        coords = inputs.unsqueeze(1)

        embedded = F.grid_sample(self.codebook, coords, mode='bilinear', padding_mode='zeros', align_corners=True).view(
            inputs.shape[0], 97, -1)
        embedded = embedded + self.position_embedding[None]

        class_embedding = self.class_embedding(class_label).unsqueeze(1)
        class_embedding = class_embedding.permute(1, 0, 2)
        embedded = embedded.permute(1, 0, 2)
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
        self.decoder = nn.Linear(embedding_size, input_size)  # For reconstructing coordinates
        self.box_regressor = nn.Linear(embedding_size, 4)  # Output box coordinates
        self.sig = nn.Sigmoid()

    def forward(self, inputs, class_label, mask=None):
        cross_output = self.encoder(inputs, class_label, mask)

        box_coordinates = self.box_regressor(cross_output[:, 0, :])

        reconstructed_coords = self.decoder(cross_output[:, 1:, :])

        box_coordinates = self.sig(box_coordinates).view(inputs.shape[0], 2, 2)
        reconstructed_coords = self.sig(reconstructed_coords)

        return reconstructed_coords, box_coordinates
