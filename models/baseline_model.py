import torch
from einops.layers.torch import Reduce

from .repeat import repeat as multi_sequential_repeat
from .perceiver_blocks import TransformerLayer
from lib.model_extra import MultiHead, ModelOutput

class BaselineModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        from lib import nomenclature

        assert self.args.n_temporal_windows == 1, f"The Baseline Model only supports one temporal window, but instead it was found {self.args.n_temporal_windows} windows"

        self.modality_to_id = {modality.name:id for id, modality in enumerate(sorted(self.args.modalities, key = lambda x: x.name))}

        self.modality_encoders = torch.nn.ModuleDict({
            modality.name: nomenclature.MODALITY_ENCODERS[modality.name](args, modality)
            for modality in self.args.modalities
        })

        self.modality_embeddings = torch.nn.Embedding(
            len(self.args.modalities), self.args.model_args.latent_dim,
        )

        self.transformer_block = multi_sequential_repeat(
            self.args.model_args.num_layers,
            lambda lnum: TransformerLayer(
                self.args,
            ),
            self.args.model_args.layer_dropout_rate,
        )

        self.classification_layer = MultiHead(args)

    def forward(self, batch, latent = None):
        all_modality_data = []
        all_modality_mask = []

        framerate_ratio = batch['video_frame_rate'] / batch['audio_frame_rate']

        for modality in self.args.modalities:
            modality_id = modality.name

            data = batch[f"modality:{modality_id}:data"]
            mask = batch[f"modality:{modality_id}:mask"]

            data = self.modality_encoders[modality_id](data, mask, framerate_ratio = framerate_ratio)

            data = data + self.modality_embeddings(torch.tensor(self.modality_to_id[modality_id]).to(data.device))

            all_modality_data.append(data)
            all_modality_mask.append(mask)

        cat_data = torch.cat(all_modality_data, dim=1)
        cat_mask = torch.cat(all_modality_mask, dim=1)

        output, _ = self.transformer_block(cat_data, cat_mask)

        output = Reduce('b n d -> b d', 'mean')(output)

        output = ModelOutput(representation = output)

        model_output = self.classification_layer(output)
        model_output['latent'] = None

        return model_output
