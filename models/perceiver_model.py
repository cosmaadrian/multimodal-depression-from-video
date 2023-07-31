import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from .perceiver_blocks import CrossAttentionBlock, SelfAttentionBlock
from lib.model_extra import MultiHead, ModelOutput

class PerceiverModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        from lib import nomenclature

        # initial random latent tensor
        self.latent = torch.nn.Parameter(
            torch.randn(
                self.args.model_args.latent_num,
                self.args.model_args.latent_dim,
            ),
        )

        # modality encoders
        self.modality_to_id = { modality.name:id for id, modality in enumerate(sorted(self.args.modalities, key = lambda x: x.name)) }

        self.modality_encoders = torch.nn.ModuleDict({
            modality.name: nomenclature.MODALITY_ENCODERS[modality.name](args, modality)
            for modality in self.args.modalities
        })

        # cross attention blocks
        if not self.args.model_args.cross_attn_parameter_sharing:
            self.cross_attn_blocks = torch.nn.ModuleDict({
                modality.name: CrossAttentionBlock(self.args)
                for modality in self.args.modalities
            })

        else:
            self.cross_attn_blocks = CrossAttentionBlock(self.args)
            # TODO clarifying details when adding modality embeddings
            self.modality_embeddings = torch.nn.Embedding(
                len(self.args.modalities), self.args.model_args.latent_dim,
            )

        # self attention blocks
        if not self.args.model_args.self_attn_parameter_sharing:
            self.self_attn_blocks = torch.nn.ModuleDict({
                modality.name: SelfAttentionBlock(self.args)
                for modality in self.args.modalities
            })
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
        for modality in self.args.modalities:
            modality_id = modality.name

            # permuting shape to iterate over temporal windows
            data = batch[f"modality:{modality_id}:data"].permute(1,0,2,3)
            mask = batch[f"modality:{modality_id}:mask"].permute(1,0,2)

            # TODO clarifying details when dealing with more than one temporal window
            for window_data, window_mask in zip(data, mask):

                # encoding modality input data
                window_data = self.modality_encoders[modality_id](window_data, window_mask)

                # applying cross attention to obtain a more enriched latent representation
                if isinstance(self.cross_attn_blocks, torch.nn.ModuleDict):
                    latent = self.cross_attn_blocks[modality_id](latent, context = window_data, mask = window_mask)
                else:
                    # TODO clarifying details when adding modality embeddings
                    # adding modality embedding when sharing the cross-attention module across modalities
                    window_data = window_data + self.modality_embeddings(torch.tensor(self.modality_to_id[modality_id]).to(window_data.device))
                    latent = self.cross_attn_blocks(latent, context = window_data, mask = window_mask)

                # applying self attention to the enriched latent representation
                if isinstance(self.self_attn_blocks, torch.nn.ModuleDict):
                    latent, _ = self.self_attn_blocks[modality_id](latent)
                else:
                    latent, _ = self.self_attn_blocks(latent)

        if self.args.model_args.extracting_embeddings:
            return latent

        # window average and final normalization
        latent = self.final_norm(
            Reduce("b n d -> b d", "mean")(latent),
        )

        # applying classification
        output = ModelOutput(representation=latent)

        return self.classification_layer(output)

