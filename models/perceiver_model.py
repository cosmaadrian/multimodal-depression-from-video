import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from perceiver_blocks import ModalityEncoderBlock, CrossAttentionBlock, TransformerBlock
from lib.model_extra import MultiHead, ModelOutput

class PerceiverModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # -- initial random latent tensor
        self.latent = nn.Parameter(
            torch.randn(
                self.args.model_args.latent_num,
                self.args.model_args.latent_dim,
            ),
        )

        # -- modality encoders
        self.encoders = torch.nn.ModuleDict()
        for modalityID in self.args.modalities:
            self.encoders[modalityID] = ModalityEncoderBlock(modalityID)

        # -- cross attention blocks
        if not self.args.model_args.cross_attn_parameter_sharing:
            self.cross_attn_blocks = torch.nn.ModuleDict()
            for modalityID in self.args.modalities:
                self.cross_attn_blocks[modalityID] = CrossAttentionBlock(self.args)
        else:
            self.cross_attn_blocks = CrossAttentionBlock(self.args)

        # -- transformer blocks
        if not self.args.model_args.transformer_parameter_sharing:
            self.transformer_blocks = torch.nn.ModuleDict()
            for modalityID in self.args.modalities:
                self.transformer_blocks[modalityID] = TransformerBlock(self.args)
        else:
            self.transformer_blocks = TransformerBlock(self.args)

        # -- classification layer
        self.classification_layer = MultiHead(args)

    def forward(self, batch, mask=None):
        # -- adding batch dimension to the latent tensor
        batch_size = batch["label"].shape[0]
        latent = repeat(self.latent, "n d -> b n d", b = batch_size)

        # -- processing the different modalities
        for modalityID in self.args.modalities:
            data = batch[f"modality:{modalityID}"]
            data = rearrange(data, "b ... d -> b (...) d")

            # -- -- encoding modality input data
            data = self.encoders[modalityID](data)

            # -- -- applying cross attention to obtain a more enriched latent representation
            if isinstance(self.cross_attn_blocks, torch.nn.ModuleDict):
                latent = self.cross_attn_blocks[modalityID](latent, context=data, mask=mask)
            else:
                latent = self.cross_attn_blocks(latent, context=data, mask=mask)

            # -- -- applying self attention to the enriched latent representation
            if isinstance(self.transformer_blocks, torch.nn.ModuleDict):
                latent = self.transformer_blocks[modalityID](latent)
            else:
                latent = self.cross_attn_blocks(latent)

        if self.args.extracting_embeddings:
            return latent

        output = ModelOutput(representation=latent)
        return self.classification_layer(output)