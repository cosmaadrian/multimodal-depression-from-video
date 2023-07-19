import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from .perceiver_blocks import TransformerLayer
from lib.model_extra import MultiHead, ModelOutput

class BaselineModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # -- modality embedding projection
        self.encoders = torch.nn.ModuleDict()
        for modalityID in self.args.modalities.keys():
            self.encoders[modalityID] = torch.nn.Linear(
                self.args.modalities[modalityID],
                self.args.model_args.attn_dim,
            )

        # -- self attention blocks
        self.self_attn_blocks = SelfAttentionBlock(self.args)

        # -- classification layer
        self.classification_layer = MultiHead(args)

    def forward(self, batch, mask=None):
        # -- adding batch dimension to the latent tensor
        batch_size = batch["labels"].shape[0]
        latent = repeat(self.latent, "n d -> b n d", b = batch_size)

        # -- processing the different modalities
        for modalityID in self.args.modalities.keys():
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
            if isinstance(self.self_attn_blocks, torch.nn.ModuleDict):
                latent = self.self_attn_blocks[modalityID](latent)
            else:
                latent = self.self_attn_blocks(latent)

        if self.args.model_args.extracting_embeddings:
            return latent
    
        # -- averaging the latent embeddings and applying classification
        output = ModelOutput(representation=latent.mean(axis=1))
        return self.classification_layer(output)
