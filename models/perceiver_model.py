import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from .perceiver_blocks import ModalityEncoderBlock, CrossAttentionBlock, SelfAttentionBlock
from lib.model_extra import MultiHead, ModelOutput

class PerceiverModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # initial random latent tensor
        self.latent = torch.nn.Parameter(
            torch.randn(
                self.args.model_args.latent_num,
                self.args.model_args.latent_dim,
            ),
        )

        # modality encoders
        self.encoders = torch.nn.ModuleDict()
        for modalityID in self.args.modalities.keys():
            self.encoders[modalityID] = ModalityEncoderBlock(modalityID, self.args)

        # cross attention blocks
        if not self.args.model_args.cross_attn_parameter_sharing:
            self.cross_attn_blocks = torch.nn.ModuleDict()
            for modalityID in self.args.modalities.keys():
                self.cross_attn_blocks[modalityID] = CrossAttentionBlock(self.args)
        else:
            self.cross_attn_blocks = CrossAttentionBlock(self.args)

        # self attention blocks
        if not self.args.model_args.self_attn_parameter_sharing:
            self.self_attn_blocks = torch.nn.ModuleDict()
            for modalityID in self.args.modalities.keys():
                self.self_attn_blocks[modalityID] = SelfAttentionBlock(self.args)
        else:
            self.self_attn_blocks = SelfAttentionBlock(self.args)

        # final normalization layer
        self.final_norm = torch.nn.LayerNorm(self.args.model_args.latent_dim)

        # classification layer
        self.classification_layer = MultiHead(args)

    def forward(self, batch):
        # adding batch dimension to the latent tensor
        batch_size = batch["labels"].shape[0]
        latent = repeat(self.latent, "n d -> b n d", b = batch_size)

        # processing the different modalities
        for modalityID in self.args.modalities.keys():
            data = batch[f"modality:{modalityID}:data"]
            mask = batch[f"modality:{modalityID}:mask"]

            # it should be done when loading the data
            # data = rearrange(data, "b ... d -> b (...) d")

            # encoding modality input data
            data = self.encoders[modalityID](data)

            # applying cross attention to obtain a more enriched latent representation
            if isinstance(self.cross_attn_blocks, torch.nn.ModuleDict):
                latent = self.cross_attn_blocks[modalityID](latent, context=data, mask=mask)
            else:
                latent = self.cross_attn_blocks(latent, context=data, mask=mask)

            # applying self attention to the enriched latent representation
            if isinstance(self.self_attn_blocks, torch.nn.ModuleDict):
                latent = self.self_attn_blocks[modalityID](latent)
            else:
                latent = self.self_attn_blocks(latent)

        if self.args.model_args.extracting_embeddings:
            return latent

        # window average and final normalization
        latent = self.final_norm(
            Reduce("b n d -> b d", "mean")(latent),
        )

        # applying classification
        output = ModelOutput(representation=latent)

        return self.classification_layer(output)

