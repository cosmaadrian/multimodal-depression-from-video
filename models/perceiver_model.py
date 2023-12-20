import torch
from einops import repeat
from einops.layers.torch import Reduce

from .perceiver_blocks import CrossAttentionBlock, SelfAttentionBlock
from lib.model_extra import MultiHead, ModelOutput

class PerceiverModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        from lib import nomenclature

        self.latent = torch.nn.Parameter(
            torch.randn(
                self.args.model_args.latent_num,
                self.args.model_args.latent_dim,
            ),
        )

        self.modality_to_id = { modality.name:id for id, modality in enumerate(sorted(self.args.modalities, key = lambda x: x.name)) }

        self.modality_encoders = torch.nn.ModuleDict({
            modality.name: nomenclature.MODALITY_ENCODERS[modality.name](args, modality)
            for modality in self.args.modalities
        })

        if not self.args.model_args.cross_attn_parameter_sharing:
            self.cross_attn_blocks = torch.nn.ModuleDict({
                modality.name: CrossAttentionBlock(self.args)
                for modality in self.args.modalities
            })
        else:
            self.cross_attn_blocks = CrossAttentionBlock(self.args)
            self.modality_embeddings = torch.nn.Embedding(
                len(self.args.modalities), self.args.model_args.latent_dim,
            )

        if not self.args.model_args.self_attn_parameter_sharing:
            self.self_attn_blocks = torch.nn.ModuleDict({
                modality.name: SelfAttentionBlock(self.args)
                for modality in self.args.modalities
            })
        else:
            self.self_attn_blocks = SelfAttentionBlock(self.args)

        self.final_norm = torch.nn.LayerNorm(self.args.model_args.latent_dim)

        self.classification_layer = MultiHead(args)

    def forward(self, batch, latent = None):
        if latent is None:
            batch_size = batch["video_frame_rate"].shape[0]
            latent = repeat(self.latent, "n d -> b n d", b = batch_size)

        framerate_ratio = batch['video_frame_rate'] / batch['audio_frame_rate']

        for modality in self.args.modalities:
            modality_id = modality.name

            data = batch[f"modality:{modality_id}:data"]
            mask = batch[f"modality:{modality_id}:mask"]

            data = self.modality_encoders[modality_id](data, mask, framerate_ratio = framerate_ratio)

            if isinstance(self.cross_attn_blocks, torch.nn.ModuleDict):
                latent = self.cross_attn_blocks[modality_id](latent, context = data, mask = mask)
            else:
                data = data + self.modality_embeddings(torch.tensor(self.modality_to_id[modality_id]).to(data.device))
                latent = self.cross_attn_blocks(latent, context = data, mask = mask)

            if isinstance(self.self_attn_blocks, torch.nn.ModuleDict):
                latent, _ = self.self_attn_blocks[modality_id](latent)
            else:
                latent, _ = self.self_attn_blocks(latent)

        avg_latent = self.final_norm(
            Reduce("b n d -> b d", "mean")(latent),
        )

        output = ModelOutput(representation=avg_latent)

        classification_output = self.classification_layer(output)
        classification_output['latent'] = latent

        return classification_output
