import torch

from .repeat import repeat as multi_sequential_repeat
from .lucidrains_perceiver import PreNorm, Attention, FeedForward

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.cross_attn = PreNorm(
            self.args.model_args.latent_dim,
            Attention(
                self.args.model_args.latent_dim,
                self.args.model_args.context_dim,
                heads = self.args.model_args.cross_attn_num_heads,
                dim_head = self.args.model_args.cross_attn_dim_head,
                dropout = self.args.model_args.dropout_rate,
            ),
            context_dim = self.args.model_args.context_dim,
        )

        self.cross_ff = PreNorm(
            self.args.model_args.latent_dim,
            FeedForward(
                self.args.model_args.latent_dim,
                dropout = self.args.model_args.dropout_rate,
            )
        )

    def forward(self, latent, context = None, mask = None):
        latent = self.cross_attn(latent, context = context, mask = mask) + latent
        latent = self.cross_ff(latent) + latent

        return latent

class SelfAttentionBlock(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if self.args.model_args.self_attn_block_type == "transformer":
            self_attn_layer = TransformerLayer
        else:
            raise NotImplementedError("Support only transformer")

        self.self_attn_block = multi_sequential_repeat(
            self.args.model_args.self_attn_num_layers,
            lambda _: self_attn_layer(
                self.args,
            ),
            self.args.model_args.layer_dropout_rate,
        )

    def forward(self, latent, mask = None):
        latent, mask = self.self_attn_block(latent, mask)
        return latent, mask

class TransformerLayer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.self_attn = PreNorm(
            self.args.model_args.latent_dim,
            Attention(
                self.args.model_args.latent_dim,
                heads = self.args.model_args.self_attn_num_heads,
                dim_head = self.args.model_args.self_attn_dim_head,
                dropout = self.args.model_args.dropout_rate,
            ),
        )

        self.self_ff = PreNorm(
            self.args.model_args.latent_dim,
            FeedForward(
                self.args.model_args.latent_dim,
                dropout = self.args.model_args.dropout_rate,
            ),
        )

    def forward(self, latent, mask = None):
        latent = self.self_attn(latent, mask = mask) + latent
        latent = self.self_ff(latent) + latent

        return latent, mask
