import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from .repeat import repeat
from .perceiver_blocks import TransformerLayer
from lib.model_extra import MultiHead, ModelOutput

class BaselineModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # sanity checking
        assert self.args.n_temporal_windows == 1, f"The Baseline Model only supports one temporal window, but instead it was found {self.args.n_temporal_windows} windows"

        # modality projection
        self.projections = torch.nn.ModuleDict()
        for modalityID in self.args.modalities.keys():
            self.projections[modalityID] = torch.nn.Linear(
                self.args.modalities[modalityID],
                self.args.model_args.latent_dim,
            )

        # positional and modality embeddings
        self.positional_embeddings = torch.nn.ModuleDict()
        self.modality_embeddings = torch.nn.ModuleDict()
        for modalityID in self.args.modalities.keys():
            max_fps = self.args.max_audio_fps if "audio" in modalityID else args.max_video_fps
            max_length = max_fps * self.args.seconds_per_window # it should be only one temporal window

            self.positional_embeddings[modalityID] = torch.nn.Embedding(
                max_length, self.args.model_args.latent_dim,
            )

            self.modality_embeddings[modalityID] = torch.nn.Embedding(
                max_length, self.args.model_args.latent_dim,
            )

            # TODO: Before projection a reshape should be done to distinguish one hand from another but ...
            if "hand" in modalityID:
                self.hands_frontier = self.args.model_args.latent_dim // 2
                # special left- and right-hand modality embeddings
                self.left_hand_embedding = torch.nn.Embedding(
                    max_length, self.hands_frontier,
                )
                self.right_hand_embedding = torch.nn.Embedding(
                    max_length, self.hands_frontier,
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
        for modalityID in self.args.modalities.keys():
            max_fps = self.args.max_audio_fps if "audio" in modalityID else self.args.max_video_fps
            max_length = max_fps * self.args.seconds_per_window # it should be only one temporal window

            # note that temporal window dimension is squeezed
            data = batch[f"modality:{modalityID}:data"].squeeze(1)
            mask = batch[f"modality:{modalityID}:mask"].squeeze(1)
            time_steps = torch.arange(max_length).to(data.device)

            # projecting input data
            data = self.projections[modalityID](data)

            # adding positional embedding
            data = data + self.positional_embeddings[modalityID](time_steps)

            # adding modality embedding
            data = data + self.modality_embeddings[modalityID](time_steps)

            # adding special left- and right-hand embeddings
            if modalityID == "hand_landmarks":
                # TODO: Discuss about this hands' issue
                data[:, :, :self.hands_frontier] = data[:, :, :self.hands_frontier] + self.left_hand_embedding(time_steps)
                data[:, :, self.hands_frontier:] = data[:, :, self.hands_frontier:] + self.right_hand_embedding(time_steps)

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
