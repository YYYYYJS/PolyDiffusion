"""
Transformer implementation adapted from CLIP ViT:
https://github.com/openai/CLIP/blob/4c0275784d6d9da97ca1f47eaaee31de1867da91/clip/model.py
"""

import math
from contour_diffusion.Encoder import TransformerEncoder
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def get_l(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def renormalize_attention(A_adjust):
    row_sum = A_adjust.sum(dim=3, keepdims=True)
    return A_adjust / row_sum


def generate_image_patch_boundary_points(resolution_to_attention):
    image_patch_boundary_points = {}

    for resolution in resolution_to_attention:
        boundary_points = []
        block_size = 1 / resolution
        interval = block_size / 24

        for i in range(resolution):
            for j in range(resolution):
                x1 = j * block_size
                y1 = i * block_size
                center_x = x1 + 0.5 * block_size
                center_y = y1 + 0.5 * block_size
                left_boundary = [(x1, y1 + y * interval) for y in range(25)]
                y1 = left_boundary[-1][1]

                bottom_boundary = [(x1 + (x + 1) * interval, y1) for x in range(24)]
                x1 = bottom_boundary[-1][0]

                right_boundary = [(x1, y1 - (y + 1) * interval) for y in range(24)]
                y1 = right_boundary[-1][1]

                top_boundary = [((x1 - (x + 1) * interval), y1) for x in range(23)]

                block_boundary_points = left_boundary + bottom_boundary + right_boundary + top_boundary
                block_boundary_points = [(x, y) for x, y in block_boundary_points]

                flat_block_boundary_points = [coord / 1 for point in block_boundary_points for coord in point]
                head = [center_x, center_y]
                flat_block_boundary_points = head + flat_block_boundary_points

                boundary_points.append(flat_block_boundary_points)

        image_patch_boundary_points[f'boundary_points_resolution{resolution}'] = boundary_points

    return image_patch_boundary_points


def xf_convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: th.Tensor):
        return super().forward(x.float()).to(x.dtype)


class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx, width, heads):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3).to(dtype=torch.float16)
        self.c_proj = nn.Linear(width, width).to(dtype=torch.float16)
        self.attention = QKVMultiheadAttention(heads, n_ctx)

    def forward(self, x, key_padding_mask=None, attention_weights=None):
        x = self.c_qkv(x)
        x = self.attention(x, key_padding_mask, attention_weights)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx

    def forward(self, qkv, key_padding_mask=None, attention_weights=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = th.split(qkv, attn_ch, dim=-1)
        weight = th.einsum("bthc,bshc->bhts", q * scale, k * scale)

        if key_padding_mask is not None:
            weight = weight.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (N, 1, 1, L1)
                float('-inf'),
            )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        if attention_weights is not None:
            a = attention_weights.repeat(self.n_heads, 1, 1).reshape(bs, self.n_heads, 10, 10)
            weight = weight * (1 + 0.2 * a)

            weight = weight.to(dtype=torch.float16)
            weight = renormalize_attention(weight)

            return th.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            n_ctx: int,
            width: int,
            heads: int,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx,
            width,
            heads,
        )
        self.ln_1 = LayerNorm(width)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x: th.Tensor, key_padding_mask=None, attention_weights=None):
        x = x + self.attn(self.ln_1(x), key_padding_mask, attention_weights)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            n_ctx: int,
            width: int,
            layers: int,
            heads: int,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx,
                    width,
                    heads,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: th.Tensor, key_padding_mask=None, attention_weights=None):
        for block in self.resblocks:
            x = block(x, key_padding_mask, attention_weights)
        return x


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units)).to(
            dtype=torch.float32)
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.cat((torch.arange(length_q // 2 + 1), torch.arange(length_q // 2 - 1, 0, -1)))
        range_vec_k = torch.cat((torch.arange(length_k // 2 + 1), torch.arange(length_k // 2 - 1, 0, -1)))
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.tensor(final_mat, dtype=torch.int)
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings


class MultiHeadRelativeAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 96

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.tensor([self.head_dim])).to('cuda', dtype=torch.float32)

    def forward(self, qkv, mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        batch_size, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        qkv = qkv.view(batch_size, n_ctx, -1)
        query, key, value = th.chunk(qkv, chunks=3, dim=2)

        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x


class MultiheadRelativeAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = MultiHeadRelativeAttention(width, heads, 0)

    def forward(self, x, key_padding_mask=None):
        x = self.c_qkv(x)
        x = self.attention(x, key_padding_mask)
        x = self.c_proj(x)
        return x


class ResidualAttentionRelativeBlock(nn.Module):
    def __init__(
            self,
            width: int,
            heads: int,
    ):
        super().__init__()

        self.attn = MultiheadRelativeAttention(
            width,
            heads

        )
        self.ln_1 = LayerNorm(width)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x: th.Tensor, key_padding_mask=None):
        x = x + self.attn(self.ln_1(x), key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class ContourTransformer(nn.Module):
    def __init__(
            self,

            width: int,
            layers: int,
            heads: int,
    ):
        super().__init__()

        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionRelativeBlock(

                    width,
                    heads,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: th.Tensor):
        for block in self.resblocks:
            x = block(x)
        return torch.mean(x, dim=1)


class ContourTransformerEncoder(nn.Module):
    def __init__(
            self,
            contour_length: int,

            output_dim: int,
            num_layers: int,
            num_heads: int,
            use_final_ln: bool,
            num_classes_for_contour_object: int,
            mask_size_for_contour_object: int,
            used_condition_types=[
                'obj_description', 'obj_contour', 'is_valid_obj', 'prompt', 'mask', 'mask_'
            ],
            hidden_dim: int = 256,
            use_positional_embedding=True,
            resolution_to_attention=[],
            use_key_padding_mask=False,
            not_use_contour_fusion_module=False
    ):
        super().__init__()
        self.not_use_contour_fusion_module = not_use_contour_fusion_module
        self.use_key_padding_mask = use_key_padding_mask
        self.used_condition_types = used_condition_types
        self.num_classes_for_contour_object = num_classes_for_contour_object
        self.mask_size_for_contour_object = mask_size_for_contour_object
        self.contour_dim = hidden_dim
        self.contour_point_num = 97
        if not self.not_use_contour_fusion_module:
            self.transform = Transformer(
                n_ctx=contour_length,
                width=hidden_dim,
                layers=num_layers,
                heads=num_heads
            )
        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(th.empty(contour_length, hidden_dim, dtype=th.float32))
        self.transformer_proj = nn.Linear(hidden_dim, output_dim)

        if 'obj_description' in self.used_condition_types:
            self.obj_description_embedding = nn.Embedding(185, hidden_dim)

        if 'obj_contour' in self.used_condition_types:
            # self.contour_embedding = nn.Linear(2, self.contour_dim)

            self.obj_contour_embedding = TransformerEncoder(2, 185, hidden_dim, hidden_dim,
                                                            num_heads, num_layers, self.contour_point_num)

            # self.xf_project = nn.Linear(hidden_dim, hidden_dim)
        if use_final_ln:
            self.final_ln = LayerNorm(hidden_dim)
        else:
            self.final_ln = None

        self.dtype = torch.float32

        self.resolution_to_attention = resolution_to_attention
        self.image_patch_contour_embedding = {}

        temp = generate_image_patch_boundary_points(resolution_to_attention)

        self.image_patch_contour_embedding['resolution8'] = torch.tensor(temp['boundary_points_resolution8']).to(
            'cuda')
        self.image_patch_contour_embedding['resolution16'] = torch.tensor(temp['boundary_points_resolution16']).to(
            'cuda')
        self.image_patch_contour_embedding['resolution32'] = torch.tensor(temp['boundary_points_resolution32']).to(
            'cuda')

    def convert_to_fp16(self):
        self.dtype = torch.float16
        # self.contour_embedding.to(th.float16)
        if not self.not_use_contour_fusion_module:
            self.transform.apply(xf_convert_module_to_f16)
        self.transformer_proj.to(th.float16)
        if self.use_positional_embedding:
            self.positional_embedding.to(th.float16)
        if 'obj_description' in self.used_condition_types:
            self.obj_description_embedding.to(th.float16)
        if 'obj_contour' in self.used_condition_types:
            self.obj_contour_embedding.apply(xf_convert_module_to_f16)
            self.obj_contour_embedding.half()

            # self.xf_project.to(th.float16)

    def forward(self, obj_description=None, obj_contour=None, obj_mask=None, is_valid_obj=None,
                obj_bbox=None,
                ):

        global mask_embedding, value, obj_class
        assert (obj_description is not None) or (obj_contour is not None) or (obj_mask is not None)
        outputs = {}
        o = 2.4142135623730951
        BSZ = obj_contour.shape[0]
        attention_weights = torch.zeros((BSZ, 10, 10), dtype=torch.float16, device='cuda')

        for k in range(BSZ):
            obj_bbox_list = obj_bbox[k].tolist()
            center_points = []
            num_obj = torch.sum(is_valid_obj[k] == 1).item()
            for i in range(0, num_obj):
                center_points.append(
                    (obj_bbox_list[i][0] + 0.5 * obj_bbox_list[i][2], obj_bbox_list[i][1] + 0.5 * obj_bbox_list[i][3]))

            for m in range(1, num_obj):
                for n in range(1, num_obj):
                    if m != n:
                        dis = get_l(center_points[m], center_points[n])
                        attention_weights[k, m, n] = o - dis
                        attention_weights[k, n, m] = o - dis

        attention_weights = attention_weights.to(torch.float16)

        xf_in = None
        if self.use_positional_embedding:
            xf_in = self.positional_embedding[None]

        if 'obj_description' in self.used_condition_types:
            obj_description = obj_description.int()
            obj_class = obj_description
            obj_description = self.obj_contour_embedding.class_embedding(obj_description)

            if xf_in is None:
                xf_in = obj_description
            else:
                xf_in = xf_in + obj_description
        outputs['obj_description_embedding'] = obj_description.permute(0, 2, 1)

        if 'obj_contour' in self.used_condition_types:

            obj_contour = obj_contour.reshape(BSZ * 10, self.contour_point_num, 2)
            obj_class = obj_class.view(-1)
            # obj_contour_embedding = self.contour_embedding(obj_contour).reshape(BSZ * 10, self.contour_point_num, -1)
            obj_contour_embedding = self.obj_contour_embedding(obj_contour.to(dtype=torch.float16), obj_class,
                                                               mask=None)[:, 0, :]
            obj_contour_embedding = obj_contour_embedding.reshape(BSZ, 10, self.contour_dim)
            if xf_in is None:
                xf_in = obj_contour_embedding
            else:

                xf_in = xf_in + obj_contour_embedding

            outputs['obj_contour_embedding'] = obj_contour_embedding.permute(0, 2, 1)

            for resolution in self.resolution_to_attention:
                B = resolution ** 2
                class_lable = torch.randn(B)
                class_lable = torch.zeros_like(class_lable).to('cuda', dtype=torch.int)
                x = self.image_patch_contour_embedding['resolution{}'.format(resolution)].to(dtype=torch.float16)
                x = x.reshape(B, self.contour_point_num, 2)
                # x = self.contour_embedding(x)
                temp = self.obj_contour_embedding(
                    x, class_lable, mask=None)[:, 0, :].unsqueeze(0).permute(0, 2, 1).repeat(BSZ, 1, 1)
                outputs[
                    'image_patch_contour_embedding_for_resolution{}'.format(resolution)] = temp
        if 'is_valid_obj' in self.used_condition_types:
            outputs['key_padding_mask'] = (1 - is_valid_obj).bool()  # (N, L2)

        key_padding_mask = outputs['key_padding_mask'] if self.use_key_padding_mask else None

        if self.not_use_contour_fusion_module:
            xf_out = xf_in.to(dtype=torch.float16)
        else:
            xf_out = self.transform(xf_in.to(dtype=torch.float16), key_padding_mask, attention_weights)  # NLC

        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, 0])  # NC
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs['xf_proj'] = xf_proj
        outputs['xf_out'] = xf_out

        return outputs


class SingDiffusionEncoder(nn.Module):
    def __init__(
            self,
            contour_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int = 128,
            use_positional_embedding=True,

    ):
        super().__init__()

        self.contour_dim = hidden_dim
        self.contour_point_num = 97

        self.transform = Transformer(
            n_ctx=contour_length,
            width=hidden_dim,
            layers=num_layers,
            heads=num_heads
        )
        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(th.empty(contour_length, hidden_dim, dtype=th.float32))

        self.obj_description_embedding = nn.Embedding(185, hidden_dim)

        self.obj_contour_embedding = TransformerEncoder(2, 185, hidden_dim, hidden_dim,
                                                        num_heads, num_layers, self.contour_point_num)

        self.dtype = torch.float32

    def forward(self, obj_description=None, obj_contour=None, is_valid_obj=None,
                obj_bbox=None,
                ):

        global mask_embedding, value, obj_class

        outputs = {}
        o = 2.4142135623730951
        BSZ = obj_contour.shape[0]

        xf_in = None
        if self.use_positional_embedding:
            xf_in = self.positional_embedding[None]

        obj_description = obj_description.int()
        obj_class = obj_description
        obj_description = self.obj_contour_embedding.class_embedding(obj_description)

        attention_weights = torch.zeros((BSZ, 10, 10), dtype=torch.float32, device='cuda')

        for k in range(BSZ):
            obj_bbox_list = obj_bbox[k].tolist()
            center_points = []
            num_obj = torch.sum(is_valid_obj[k] == 1).item()
            for i in range(0, num_obj):
                center_points.append(
                    (obj_bbox_list[i][0] + 0.5 * obj_bbox_list[i][2], obj_bbox_list[i][1] + 0.5 * obj_bbox_list[i][3]))

            class_similarity = (F.cosine_similarity(obj_description[k].unsqueeze(1), obj_description[k].unsqueeze(0),
                                                    dim=2) / 2) + 0.5
            for m in range(1, num_obj):
                for n in range(1, num_obj):
                    if m != n:
                        dis = get_l(center_points[m], center_points[n])
                        attention_weights[k, m, n] = (o - dis) / 2 + class_similarity[m][n] / 2
                        attention_weights[k, n, m] = (o - dis) / 2 + class_similarity[n][m] / 2

        attention_weights = attention_weights.to(torch.float32)

        if xf_in is None:
            xf_in = obj_description
        else:
            xf_in = xf_in + obj_description
        outputs['obj_description_embedding'] = obj_description.permute(0, 2, 1)

        obj_contour = obj_contour.reshape(BSZ * 10, self.contour_point_num, 2)
        obj_class = obj_class.view(-1)
        # obj_contour_embedding = self.contour_embedding(obj_contour).reshape(BSZ * 10, self.contour_point_num, -1)
        obj_contour_embedding = self.obj_contour_embedding(obj_contour.to(dtype=torch.float32), obj_class,
                                                           mask=None)[:, 0, :]
        obj_contour_embedding = obj_contour_embedding.reshape(BSZ, 10, self.contour_dim)
        if xf_in is None:
            xf_in = obj_contour_embedding
        else:

            xf_in = xf_in + obj_contour_embedding

        outputs['obj_contour_embedding'] = obj_contour_embedding.permute(0, 2, 1)

        outputs['key_padding_mask'] = (1 - is_valid_obj).bool()  # (N, L2)

        key_padding_mask = outputs['key_padding_mask']

        xf_out = self.transform(xf_in.to(dtype=torch.float32), key_padding_mask, attention_weights)  # NLC
        xf_out = xf_out.to(torch.float32)

        return xf_out
