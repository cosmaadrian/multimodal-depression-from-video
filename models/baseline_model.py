import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from .repeat import repeat
from .perceiver_blocks import TransformerLayer
from lib.model_extra import MultiHead, ModelOutput

from .modality_encoders import NoOpEncoder, HandLandmarkEncoder, LandmarkEncoder

class BaselineModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        from lib import nomenclature

        # sanity checking
        assert self.args.n_temporal_windows == 1, f"The Baseline Model only supports one temporal window, but instead it was found {self.args.n_temporal_windows} windows"

        self.modality_to_id = { modality.name:id for id, modality in enumerate(sorted(self.args.modalities, key = lambda x: x.name)) }

        self.modality_encoders = torch.nn.ModuleDict({
            modality.name: nomenclature[modality.name](args, modality)
            for modality in self.args.modalities
        })

        self.modality_embeddings = torch.nn.Embedding(
            len(self.args.modalities), self.args.model_args.latent_dim,
        )

        # transformer block
        self.transformer_block = repeat(
            self.args.model_args.num_layers,
            lambda lnum: TransformerLayer(
                self.args,
            ),
            self.args.model_args.layer_dropout_rate,
        )

        # final normalization layer
        self.final_norm = torch.nn.LayerNorm(self.args.model_args.latent_dim)

        # classification layer
        self.classification_layer = MultiHead(args)

    def forward(self, batch):
        # processing the different modalities
        all_modality_data = []
        all_modality_mask = []

        for modality in self.args.modalities:
            modality_id = modality.name

            # note that temporal window dimension is squeezed
            data = batch[f"modality:{modality_id}:data"].squeeze(1)
            mask = batch[f"modality:{modality_id}:mask"].squeeze(1)

            # Pre-modelling modality
            data = self.modality_encoders[modality_id](data, mask)

            # adding modality specific embedding
            data = data + self.modality_embeddings(torch.tensor(self.modality_to_id[modality_id]).to(data.device))

            all_modality_data.append(data)
            all_modality_mask.append(mask)

        # concatenating all modalities
        cat_data = torch.cat(all_modality_data, dim=1)
        cat_mask = torch.cat(all_modality_mask, dim=1)

        # applying transformer encoder
        output, _ = self.transformer_block(cat_data, cat_mask)

        if self.args.model_args.extracting_embeddings:
            return output

        # window average and final normalization
        output = self.final_norm(
            Reduce('b n d -> b d', 'mean')(output),
        )

        # applying classification
        output = ModelOutput(representation = output)

        return self.classification_layer(output)
