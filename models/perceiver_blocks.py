import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from lucidrains_perceiver import PreNorm, Attention, FeedForward

class ModalityEncoderBlock(torch.nn.Module):
    def __init__(self, modalityID, args):
        super().__init__()
        self.args = args
        self.encoder_dim = self.args.MODALITY_FRONTEND_INPUT_SIZE[modalityID]
        
        self.encoder_block = torch.nn.Linear(
            self.encoder_dim,
            self.args.model_args.input_dim
        )

        # TODO: Implement Positional Encoding via PyTorch Embeddings (ask Adrian)

    def forward(self, data):

        return self.encoder_block(data)

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # -- cross attention module
        self.cross_attn = PreNorm(
            self.args.model_args.latent_dim,
            Attention(
                self.args.model_args.latent_dim,
                self.args.model_args.input_dim,
                heads=self.args.model_args.cross_attn_num_heads,
                dim_head=self.args.model_args.cross_attn_dim_head,
                dropout=self.args.model_args.dropout_rate,
            ),
            context_dim = self.args.model_args.input_dim,
        )

        # -- feed forward module
        self.cross_ff = PreNorm(
            self.args.model_args.latent_dim,
            FeedForward(
                self.args.mdoel_args.latent_dim,
                dropout=self.args.model_args.dropout_rate,
            )
        )

    def forward(self, latent, data, mask = None):

        latent = self.cross_attn(latent, context=data, mask=mask) + latent
        latent = self.cross_ff(latent) + latent

        return x

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # -- cross attention module
        self.cross_attn = PreNorm(
            self.args.model_args.latent_dim,
            Attention(
                self.args.model_args.latent_dim,
                self.args.model_args.input_dim,
                heads=self.args.model_args.cross_attn_num_heads,
                dim_head=self.args.model_args.cross_attn_dim_head,
                dropout=self.args.model_args.dropout_rate,
            ),
            context_dim = self.args.model_args.input_dim,
        )

        # -- feed forward module
        self.cross_ff = PreNorm(
            self.args.model_args.latent_dim,
            FeedForward(
                self.args.mdoel_args.latent_dim,
                dropout=self.args.model_args.dropout_rate,
            )
        )

    def forward(self, latent, data, mask = None):

        latent = self.cross_attn(latent, context=data, mask=mask) + latent
        latent = self.cross_ff(latent) + latent

        return x

class TransformerBlock(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.transformer_block = torch.nn.Sequential(
            *[TransformerLayer(self.args) for _ in range(self.args.transformer_num_layers)],
        )

    def forward(self, latent):
        
        latent = self.transformer_block(latent)

        return latent
    
class TransformerLayer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # -- self attention module
        self.self_attn = PreNorm(
            self.args.model_args.latent_dim,
            Attention(
                self.args.model_args.latent_dim,
                heads=self.args.model_args.transformer_num_heads,
                dim_head=self.args.model_args.transformer_dim_head,
                dropout=self.args.model_args.dropout_rate,
            ),
        )

        # -- feed forward module
        self.self_ff = PreNorm(
            self.args.model_args.latent_dim,
            FeedForward(
                self.args.model_args.latent_dim,
                dropout=self.args.model_args.dropout_rate,
            ),
        )

    def forward(self, latent):

        latent = self.self_attn(latent) + latent
        latent = self.self_ff(latent) + latent
            
        return latent