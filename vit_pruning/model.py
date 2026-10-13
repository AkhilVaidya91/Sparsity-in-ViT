"""Masked Vision Transformer components."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ATTENTION_HEADS, DROPOUT, HIDDEN_SIZE, MLP_DIMENSION, NUM_CHANNELS,
    NUM_CLASSES, NUM_PATCHES, PATCH_SIZE, TRANSFORMER_BLOCKS,
)


class MaskedLinear(nn.Linear):
    """
    Linear layer with a fixed unstructured parameter mask.

    The underlying parameter tensors remain trainable, but masked
    entries are excluded from the forward pass. The optimizer step
    is followed by explicit mask enforcement so pruned entries remain
    exactly zero in storage as well.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

        self.register_buffer(
            "weight_mask",
            torch.ones_like(self.weight),
        )

        if self.bias is not None:
            self.register_buffer(
                "bias_mask",
                torch.ones_like(self.bias),
            )

    def forward(self, input):
        masked_weight = self.weight * self.weight_mask

        if self.bias is None:
            masked_bias = None
        else:
            masked_bias = self.bias * self.bias_mask

        return F.linear(
            input,
            masked_weight,
            masked_bias,
        )


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_channels=NUM_CHANNELS,
            out_channels=HIDDEN_SIZE,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MaskedMultiheadAttention(nn.MultiheadAttention):
    """
    Standard PyTorch MHA with elementwise unstructured parameter masks.

    The pruning mask covers:
      * in_proj_weight
      * in_proj_bias
      * out_proj.weight
      * out_proj.bias
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        batch_first=True,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.register_buffer(
            "in_proj_weight_mask",
            torch.ones_like(self.in_proj_weight),
        )

        if self.in_proj_bias is not None:
            self.register_buffer(
                "in_proj_bias_mask",
                torch.ones_like(self.in_proj_bias),
            )

        original_out_proj = self.out_proj

        self.out_proj = MaskedLinear(
            embed_dim,
            embed_dim,
            bias=(original_out_proj.bias is not None),
        )

        with torch.no_grad():
            self.out_proj.weight.copy_(original_out_proj.weight)

            if self.out_proj.bias is not None:
                self.out_proj.bias.copy_(original_out_proj.bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        masked_in_proj_weight = (
            self.in_proj_weight * self.in_proj_weight_mask
        )

        if self.in_proj_bias is None:
            masked_in_proj_bias = None
        else:
            masked_in_proj_bias = (
                self.in_proj_bias * self.in_proj_bias_mask
            )

        masked_out_proj_weight = (
            self.out_proj.weight * self.out_proj.weight_mask
        )

        if self.out_proj.bias is None:
            masked_out_proj_bias = None
        else:
            masked_out_proj_bias = (
                self.out_proj.bias * self.out_proj.bias_mask
            )

        attn_output, attn_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=masked_in_proj_weight,
            in_proj_bias=masked_in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=masked_out_proj_weight,
            out_proj_bias=masked_out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(HIDDEN_SIZE)
        self.layer_norm2 = nn.LayerNorm(HIDDEN_SIZE)

        self.multihead_attention = MaskedMultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=ATTENTION_HEADS,
            batch_first=True,
            dropout=dropout,
        )

        self.mlp = nn.Sequential(
            MaskedLinear(HIDDEN_SIZE, MLP_DIMENSION),
            nn.GELU(),
            nn.Dropout(dropout),
            MaskedLinear(MLP_DIMENSION, HIDDEN_SIZE),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm1(x)

        attention_output, _ = self.multihead_attention(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )

        x = x + attention_output

        x_norm = self.layer_norm2(x)

        mlp_output = self.mlp[0](x_norm)
        mlp_output = self.mlp[1](mlp_output)
        mlp_output = self.mlp[2](mlp_output)

        mlp_output = self.mlp[3](mlp_output)
        mlp_output = self.mlp[4](mlp_output)

        x = x + mlp_output

        return x


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()

        self.patch_embedding = PatchEmbedding()

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                HIDDEN_SIZE,
            )
        )

        self.position_embeddings = nn.Parameter(
            torch.randn(
                1,
                NUM_PATCHES + 1,
                HIDDEN_SIZE,
            )
        )

        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerEncoder(dropout)
                for _ in range(TRANSFORMER_BLOCKS)
            ]
        )

        self.mlp_head = MLPHead()

    def forward(self, x):
        x = self.patch_embedding(x)

        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            (cls_tokens, x),
            dim=1,
        )

        x = x + self.position_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.mlp_head(x[:, 0, :])

        return x
