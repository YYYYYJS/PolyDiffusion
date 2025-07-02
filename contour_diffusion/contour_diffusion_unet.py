import math
from abc import abstractmethod

import cv2
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding

from repositories.dpm_solver.example_pytorch.models.guided_diffusion.unet import QKVAttention
from .fp16_util import convert_module_to_f16
from .nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .util import get_obj_from_str


class LogisticSigmoidFunction(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(70 * (x - 0.775))


# 创建模型
logistic_sigmoid_model = LogisticSigmoidFunction()


def build_model(cfg):
    contour_encoder = get_obj_from_str(cfg.model.parameters.contour_encoder.type)(
        contour_length=cfg.data.parameters.contour_length,
        num_classes_for_contour_object=cfg.data.parameters.num_classes_for_contour_object,
        mask_size_for_contour_object=cfg.data.parameters.mask_size_for_contour_object,
        **cfg.model.parameters.contour_encoder.parameters
    )

    model_kwargs = dict(**cfg.model.parameters)
    model_kwargs.pop('contour_encoder')
    return get_obj_from_str(cfg.model.type)(
        contour_encoder=contour_encoder,
        **model_kwargs,
    )


class SiLU(nn.Module):  # export-friendly version of SiLU()
    @staticmethod
    def forward(x):
        return x * th.sigmoid(x)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cond_kwargs=None):
        extra_output = None

        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, (AttentionBlock, ObjectSpacePerceptionCrossAttention)):
                x, extra_output = layer(x, cond_kwargs)
            else:
                x = layer(x)
        return x, extra_output


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            encoder_channels=None,
            return_attention_embeddings=False,
            ds=None,
            resolution=None,
            type=None,
            use_positional_embedding=False
    ):
        super().__init__()
        self.type = type
        self.ds = ds
        self.resolution = resolution
        self.return_attention_embeddings = return_attention_embeddings

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(
                th.randn(channels // self.num_heads, resolution ** 2) / channels ** 0.5)  # [C,L1]
        else:
            self.positional_embedding = None

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)

        self.qkv = conv_nd(1, channels, channels * 3, 1)

        self.attention = QKVAttentionLegacy(self.num_heads)

        self.encoder_channels = encoder_channels
        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond_kwargs=None):
        '''
        :param x: (N, C, H, W)
        :param cond_kwargs['xf_out']: (N, C, L2)
        :return:
            extra_output: N x L2 x 3 x ds x ds
        '''
        extra_output = None
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # N x C x (HxW)

        qkv = self.qkv(self.norm(x))  # N x 3C x L1, 其中L1=H*W
        if cond_kwargs is not None and self.encoder_channels is not None:
            kv_for_encoder_out = self.encoder_kv(
                cond_kwargs['xf_out'])  # xf_out: (N x encoder_channels x L2) -> (N x 2C x L2), 其中L2=max_obj_num
            h = self.attention(qkv, kv_for_encoder_out, positional_embedding=self.positional_embedding)
        else:
            h = self.attention(qkv, positional_embedding=self.positional_embedding)
        h = self.proj_out(h)
        output = (x + h).reshape(b, c, *spatial)

        if self.return_attention_embeddings:
            assert cond_kwargs is not None
            if extra_output is None:
                extra_output = {}
            extra_output.update({
                'type': self.type,
                'ds': self.ds,
                'resolution': self.resolution,
                'num_heads': self.num_heads,
                'num_channels': self.channels,
                'image_query_embeddings': qkv[:, :self.channels, :].detach(),  # N x C x L1
            })
            if cond_kwargs is not None:
                extra_output.update({
                    'layout_key_embeddings': kv_for_encoder_out[:, : self.channels, :].detach()  # N x C x L2
                })

        return output, extra_output


class ObjectSpacePerceptionCrossAttention(nn.Module):

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            encoder_channels=None,
            return_attention_embeddings=False,
            ds=None,
            resolution=None,
            type=None,
            use_positional_embedding=True,
            use_key_padding_mask=False,
            channels_scale_for_positional_embedding=1.0,
            norm_first=False,
            norm_for_obj_embedding=False
    ):
        super().__init__()
        self.norm_for_obj_embedding = None
        self.norm_first = norm_first
        self.channels_scale_for_positional_embedding = channels_scale_for_positional_embedding
        self.use_key_padding_mask = use_key_padding_mask
        self.type = type
        self.ds = ds
        self.resolution = resolution
        self.return_attention_embeddings = return_attention_embeddings

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_positional_embedding = use_positional_embedding
        # assert self.use_positional_embedding

        self.use_checkpoint = use_checkpoint

        self.self_qkv_projector = conv_nd(1, channels, 3 * channels, 1)
        self.cross_q_projector = conv_nd(1, channels, channels, 1)
        self.norm_for_qkv = normalization(channels)
        if encoder_channels is not None:
            self.encoder_channels = encoder_channels
            self.contour_content_embedding_projector = conv_nd(1, encoder_channels, 2 * channels, 1)
            self.obj_class_embedding_projector = conv_nd(1, encoder_channels, channels, 1)
            self.contour_position_embedding_projector = conv_nd(1, encoder_channels,
                                                                int(channels * self.channels_scale_for_positional_embedding),
                                                                1)
            if self.norm_first:
                if norm_for_obj_embedding:
                    self.norm_for_obj_embedding = normalization(encoder_channels)
                self.norm_for_obj_description_embedding = normalization(encoder_channels)
                self.norm_for_contour_positional_embedding = normalization(encoder_channels)
                self.norm_for_image_patch_positional_embedding = normalization(encoder_channels)
            else:
                self.norm_for_obj_description_embedding = normalization(encoder_channels)

                self.norm_for_contour_positional_embedding = normalization(
                    int(channels * self.channels_scale_for_positional_embedding))
                self.norm_for_image_patch_positional_embedding = normalization(
                    int(channels * self.channels_scale_for_positional_embedding))

        self.self_proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.cross_proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond_kwargs):
        '''
        :param x: (N, C, H, W)
        :param cond_kwargs['xf_out']: (N, C, L2)
        :return:
            extra_output: N x L2 x 3 x ds x ds
        '''
        extra_output = None
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # N x C x (HxW)

        self_qkv = self.self_qkv_projector(self.norm_for_qkv(x))  # N x 3C x L1, 其中L1=H*W
        bs, C, L1, L2 = self_qkv.shape[0], self.channels, self_qkv.shape[2], cond_kwargs['obj_contour_embedding'].shape[
            -1]

        # positional embedding for image patch

        image_patch_positional_embedding = self.contour_position_embedding_projector(
            cond_kwargs['image_patch_contour_embedding_for_resolution{}'.format(self.resolution)]
        )
        image_patch_positional_embedding = self.norm_for_image_patch_positional_embedding(
            image_patch_positional_embedding)
        image_patch_positional_embedding = image_patch_positional_embedding.reshape(bs * self.num_heads,
                                                                                    int(C * self.channels_scale_for_positional_embedding) // self.num_heads,
                                                                                    L1)

        # content embedding for image patch
        self_q_image_patch_content_embedding, self_k_image_patch_content_embedding, self_v_image_patch_content_embedding = self_qkv.split(
            C,
            dim=1)
        self_q_image_patch_content_embedding = self_q_image_patch_content_embedding.reshape(bs * self.num_heads,
                                                                                            C // self.num_heads,
                                                                                            L1)
        self_k_image_patch_content_embedding = self_k_image_patch_content_embedding.reshape(bs * self.num_heads,
                                                                                            C // self.num_heads,
                                                                                            L1)
        self_v_image_patch_content_embedding = self_v_image_patch_content_embedding.reshape(bs * self.num_heads,
                                                                                            C // self.num_heads,
                                                                                            L1)

        # embedding for image patch
        self_q_image_patch = torch.cat([self_q_image_patch_content_embedding, image_patch_positional_embedding],
                                       dim=1)
        self_k_image_patch = torch.cat([self_k_image_patch_content_embedding, image_patch_positional_embedding],
                                       dim=1)
        self_v_image_patch = self_v_image_patch_content_embedding

        scale = 1 / math.sqrt(math.sqrt(int((1 + self.channels_scale_for_positional_embedding) * C) // self.num_heads))
        self_attn_output_weights = th.einsum(
            "bct,bcs->bts", self_q_image_patch * scale, self_k_image_patch * scale

        )
        # print(f'self_map----shape=={self_attn_output_weights.shape}----heads=={self.num_heads}----batchsize=={bs}')
        self_attn_output_weights = self_attn_output_weights.view(bs * self.num_heads, L1, L1)

        self_attn_output_weights = th.softmax(self_attn_output_weights.float(), dim=-1).type(
            self_attn_output_weights.dtype)  # (N x num_heads, L1, L1+L2)

        self_attn_output = th.einsum("bts,bcs->bct", self_attn_output_weights,
                                     self_v_image_patch)  # (N x num_heads, C // num_heads, L1)
        self_attn_output = self_attn_output.reshape(bs, C, L1)  # (N, C, L1)

        self_out = self.self_proj_out(self_attn_output)
        x = (x + self_out)

        # compute cross attention--------------------------------------------------------------------------------------------------------------

        cross_q_image_patch_content_embedding = self.cross_q_projector(self.norm_for_qkv(x))

        cross_q_image_patch_content_embedding = cross_q_image_patch_content_embedding.reshape(bs * self.num_heads,
                                                                                              C // self.num_heads,
                                                                                              L1)

        # embedding for image patch
        cross_q = torch.cat([cross_q_image_patch_content_embedding, image_patch_positional_embedding],
                            dim=1)

        contour_positional_embedding = self.contour_position_embedding_projector(
            cond_kwargs['obj_contour_embedding'])
        contour_positional_embedding = self.norm_for_contour_positional_embedding(
            contour_positional_embedding)
        contour_positional_embedding = contour_positional_embedding.reshape(bs * self.num_heads,
                                                                            int(C * self.channels_scale_for_positional_embedding) // self.num_heads,
                                                                            L2)

        # content embedding for contour

        contour_content_embedding = (cond_kwargs['xf_out'] + self.norm_for_obj_description_embedding(
            cond_kwargs['obj_description_embedding'])) / 2

        k_contour_content_embedding, v_contour_content_embedding = self.contour_content_embedding_projector(
            contour_content_embedding).split(C, dim=1)  # 2 x (N x C x L2)
        k_contour_content_embedding = k_contour_content_embedding.reshape(bs * self.num_heads, C // self.num_heads,
                                                                          L2)  # (N // num_heads, C // num_heads, L2)
        v_contour_content_embedding = v_contour_content_embedding.reshape(bs * self.num_heads, C // self.num_heads,
                                                                          L2)  # (N // num_heads, C // num_heads, L2)

        # embedding for layout
        cross_k = torch.cat([k_contour_content_embedding, contour_positional_embedding],
                            dim=1)
        cross_v = v_contour_content_embedding

        scale = 1 / math.sqrt(math.sqrt(int((1 + self.channels_scale_for_positional_embedding) * C) // self.num_heads))
        cross_attn_output_weights = th.einsum(
            "bct,bcs->bts", cross_q * scale, cross_k * scale
        )
        # print(f'cross_map----shape=={cross_attn_output_weights.shape}----heads=={self.num_heads}----batchsize=={bs}')
        cross_attn_output_weights = cross_attn_output_weights.view(bs, self.num_heads, L1, L2)

        cross_attn_output_weights = cross_attn_output_weights.view(bs * self.num_heads, L1, L2)

        cross_attn_output_weights = th.softmax(cross_attn_output_weights.float(), dim=-1).type(
            cross_attn_output_weights.dtype)  # (N x num_heads, L1, L1+L2)

        cross_attn_output = th.einsum("bts,bcs->bct", cross_attn_output_weights,
                                      cross_v)  # (N x num_heads, C // num_heads, L1)
        cross_attn_output = cross_attn_output.reshape(bs, C, L1)  # (N, C, L1)

        #
        cross_out = self.cross_proj_out(cross_attn_output)

        output = (x + cross_out).reshape(b, c, *spatial)
        use_dice = True
        if use_dice:

            cross_attn_output_weights = cross_attn_output_weights.view(bs, self.num_heads, L1, L2)
            cross_attn_output_weights = torch.mean(cross_attn_output_weights, dim=1)

            if f'cross_{int(cross_attn_output_weights.shape[1] ** 0.5)}_map' in cond_kwargs:
                cond_kwargs[f'cross_{int(cross_attn_output_weights.shape[1] ** 0.5)}_layers'] = cond_kwargs[
                                                                                                    f'cross_{int(cross_attn_output_weights.shape[1] ** 0.5)}_layers'] + 1
                cond_kwargs[f'cross_{int(cross_attn_output_weights.shape[1] ** 0.5)}_map'] = cond_kwargs[
                                                                                                 f'cross_{int(cross_attn_output_weights.shape[1] ** 0.5)}_map'] + cross_attn_output_weights


            else:
                cond_kwargs[f'cross_{int(cross_attn_output_weights.shape[1] ** 0.5)}_map'] = cross_attn_output_weights
                cond_kwargs[f'cross_{int(cross_attn_output_weights.shape[1] ** 0.5)}_layers'] = 1
        return output, extra_output


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            generator,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None, positional_embedding=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Q_T, K_T, and V_T.
        :param encoder_kv: an [N x (H * 2 * C) x S] tensor of K_E, and V_E.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        if positional_embedding is not None:
            q = q + positional_embedding[None, :, :].to(q.dtype)  # [N, C, T]
            k = k + positional_embedding[None, :, :].to(q.dtype)  # [N, C, T]

        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class ContourDiffusionUNetModel(nn.Module):
    """
    A UNetModel that conditions on layout with an encoding transformer.
    The full UNet generator with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the generator.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_ds: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.

    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param {
        layout_length: number of layout objects to expect.
        hidden_dim: width of the transformer.
        num_layers: depth of the transformer.
        num_heads: heads in the transformer.
        xf_final_ln: use a LayerNorm after the output layer.
        num_classes_for_layout_object: num of classes for layout object.
        mask_size_for_layout_object: mask size for layout object image.
    }

    """

    def __init__(
            self,
            contour_encoder,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_ds,
            encoder_channels=None,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_positional_embedding_for_attention=False,
            image_size=256,
            attention_block_type='GLIDE',
            num_attention_blocks=1,
            use_key_padding_mask=False,
            channels_scale_for_positional_embedding=1.0,
            norm_first=False,
            norm_for_obj_embedding=False
    ):
        super().__init__()
        self.norm_for_obj_embedding = norm_for_obj_embedding
        self.channels_scale_for_positional_embedding = channels_scale_for_positional_embedding
        self.norm_first = norm_first
        self.use_key_padding_mask = use_key_padding_mask
        self.num_attention_blocks = num_attention_blocks
        self.attention_block_type = attention_block_type
        if self.attention_block_type == 'GLIDE':
            attention_block_fn = AttentionBlock
        elif self.attention_block_type == 'ObjectSpacePerceptionCrossAttention':
            attention_block_fn = ObjectSpacePerceptionCrossAttention

        self.image_size = image_size
        self.use_positional_embedding_for_attention = use_positional_embedding_for_attention

        self.contour_encoder = contour_encoder

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_ds = attention_ds
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        # self.time_embed = nn.Sequential(
        #    linear(model_channels, time_embed_dim),
        #    SiLU(),
        #    linear(time_embed_dim, time_embed_dim),
        # )

        self.time_embedding = TimestepEmbedding(
            model_channels,
            time_embed_dim,
            act_fn="silu",
            cond_proj_dim=time_embed_dim
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 3, ch, 3, padding=1))]
        )

        self._feature_size = ch

        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_ds:
                    # print('encoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    for _ in range(self.num_attention_blocks):
                        layers.append(
                            attention_block_fn(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                encoder_channels=encoder_channels,
                                ds=ds,
                                resolution=int(self.image_size // ds),
                                type='input',
                                use_positional_embedding=self.use_positional_embedding_for_attention,
                                use_key_padding_mask=self.use_key_padding_mask,
                                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                                norm_first=self.norm_first,
                                norm_for_obj_embedding=self.norm_for_obj_embedding
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # print('middle attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            attention_block_fn(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
                ds=ds,
                resolution=int(self.image_size // ds),
                type='middle',
                use_positional_embedding=self.use_positional_embedding_for_attention,
                use_key_padding_mask=self.use_key_padding_mask,
                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                norm_first=self.norm_first,
                norm_for_obj_embedding=self.norm_for_obj_embedding
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_ds:
                    # print('decoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    for _ in range(self.num_attention_blocks):
                        layers.append(
                            attention_block_fn(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                encoder_channels=encoder_channels,
                                ds=ds,
                                resolution=int(self.image_size // ds),
                                type='output',
                                use_positional_embedding=self.use_positional_embedding_for_attention,
                                use_key_padding_mask=self.use_key_padding_mask,
                                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                                norm_first=self.norm_first,
                                norm_for_obj_embedding=self.norm_for_obj_embedding
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

        self.use_fp16 = use_fp16

    def convert_to_fp16(self):
        """
        Convert the torso of the generator to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        # self.out.apply(convert_module_to_f16)

        self.contour_encoder.convert_to_fp16()

    def forward(self, x, timesteps, obj_description=None, obj_contour=None, obj_mask=None, is_valid_obj=None,
                obj_bbox=None,
                time_cond=None,
                **kwargs):
        hs, extra_outputs = [], []
        bs = x.shape[0]
        # emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb = self.time_embedding(timestep_embedding(timesteps, self.model_channels), time_cond)
        contour_outputs = self.contour_encoder(
            obj_description=obj_description,
            obj_contour=obj_contour,
            obj_mask=obj_mask,
            is_valid_obj=is_valid_obj,
            obj_bbox=obj_bbox,
        )

        xf_proj, xf_out = contour_outputs["xf_proj"], contour_outputs["xf_out"]

        emb = emb + xf_proj.to(emb)

        hidden = x.type(self.dtype)
        for module in self.input_blocks:
            hidden, extra_output = module(hidden, emb, contour_outputs)

            if extra_output is not None:
                extra_outputs.append(extra_output)

            hs.append(hidden)

        hidden, extra_output = self.middle_block(hidden, emb, contour_outputs)

        if extra_output is not None:
            extra_outputs.append(extra_output)

        for module in self.output_blocks:
            hidden = th.cat([hidden, hs.pop()], dim=1)
            hidden, extra_output = module(hidden, emb, contour_outputs)

            if extra_output is not None:
                extra_outputs.append(extra_output)

        hidden = hidden.type(x.dtype)
        hidden = self.out(hidden)
        cross_images = []
        use_dice = True
        if use_dice:
            cross_32_map = contour_outputs['cross_32_map'] / contour_outputs['cross_32_layers']
            cross_16_map = contour_outputs['cross_16_map'] / contour_outputs['cross_16_layers']
            cross_8_map = contour_outputs['cross_8_map'] / contour_outputs['cross_8_layers']
            cross_32_map = cross_32_map.reshape(bs, 32, 32, 10)
            cross_16_map = cross_16_map.reshape(bs, 16, 16, 10)
            cross_8_map = cross_8_map.reshape(bs, 8, 8, 10)
            is_valid_obj = torch.sum(is_valid_obj, dim=1)
            H = hidden.shape[2]

            for i in range(bs):
                for j in range(1, int(is_valid_obj[i])):
                    image_8 = cross_8_map[i, :, :, j].to(dtype=torch.float32)
                    # 调整大小为 128x128
                    image_8 = F.interpolate(image_8.unsqueeze(0).unsqueeze(0), size=(H, H), mode='bicubic',
                                            align_corners=False)

                    image_8 = image_8 / image_8.max()

                    image_16 = cross_16_map[i, :, :, j].to(dtype=torch.float32)

                    # 调整大小为 128x128
                    image_16 = F.interpolate(image_16.unsqueeze(0).unsqueeze(0), size=(H, H), mode='bicubic',
                                             align_corners=False)

                    image_16 = image_16 / image_16.max()

                    image_32 = cross_32_map[i, :, :, j].to(dtype=torch.float32)

                    image_32 = F.interpolate(image_32.unsqueeze(0).unsqueeze(0), size=(H, H), mode='bicubic',
                                             align_corners=False)

                    image_32 = image_32 / image_32.max()

                    image = (image_8 * 0.6 + image_16 * 0.3 + image_32 * 0.1).view(1, H, H)

                    image = logistic_sigmoid_model(image).view(1, H, H)

                    '''
                    image_ = image.to('cpu').detach().numpy().squeeze()

                    # 将图像从[0, 1]转换为[0, 255]
                    image_ = (image_ * 255).astype(np.uint8)

                    cv2.imwrite(f'G:/mask/{i}__{j - 1}.png', image_)

                    '''

                    cross_images.append(image)

            if len(cross_images) != 0:
                cross_images = torch.cat(cross_images, dim=0)

        return [hidden, extra_outputs, cross_images]
