import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from .lucidrains_perceiver import PreNorm, Attention, FeedForward

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # cross attention module
        self.cross_attn = PreNorm(
            self.args.model_args.latent_dim,
            Attention(
                self.args.model_args.latent_dim,
                self.args.model_args.context_dim,
                heads=self.args.model_args.cross_attn_num_heads,
                dim_head=self.args.model_args.cross_attn_dim_head,
                dropout=self.args.model_args.dropout_rate,
            ),
            context_dim = self.args.model_args.context_dim,
        )

        # feed forward module
        self.cross_ff = PreNorm(
            self.args.model_args.latent_dim,
            FeedForward(
                self.args.model_args.latent_dim,
                dropout=self.args.model_args.dropout_rate,
            )
        )

    def forward(self, latent, context=None, mask=None):

        latent = self.cross_attn(latent, context=context, mask=mask) + latent
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

        # TODO: Use MultiSequential to allow multiple inputs and outputs in the foward pass
        self.self_attn_block = torch.nn.Sequential(
            *[self_attn_layer(self.args) for _ in range(self.args.model_args.self_attn_num_layers)],
        )

    def forward(self, latent, mask=None):

        latent = self.self_attn_block(latent, mask=mask)

        return latent, mask

class TransformerLayer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # self attention module
        self.self_attn = PreNorm(
            self.args.model_args.latent_dim,
            Attention(
                self.args.model_args.latent_dim,
                heads=self.args.model_args.self_attn_num_heads,
                dim_head=self.args.model_args.self_attn_dim_head,
                dropout=self.args.model_args.dropout_rate,
            ),
        )

        # feed forward module
        self.self_ff = PreNorm(
            self.args.model_args.latent_dim,
            FeedForward(
                self.args.model_args.latent_dim,
                dropout=self.args.model_args.dropout_rate,
            ),
        )

    def forward(self, latent, mask=None):

        latent = self.self_attn(latent, mask=mask) + latent
        latent = self.self_ff(latent) + latent

        return latent, mask
