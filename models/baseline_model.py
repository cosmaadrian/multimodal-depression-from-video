import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from .perceiver_blocks import TransformerLayer
from lib.model_extra import MultiHead, ModelOutput

class BaselineModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # -- modality projection
        self.projections = torch.nn.ModuleDict()
        for modalityID in self.args.modalities.keys():
            self.projections[modalityID] = torch.nn.Linear(
                self.args.modalities[modalityID],
                self.args.model_args.latent_dim,
            )

        # -- positional and modality embeddings
        self.positional_embeddings = torch.nn.ModuleDict()
        self.modality_embeddings = torch.nn.ModuleDict()
        for modalityID in self.args.modalities.keys():
            max_fps = self.args.max_audio_fps if "audio" in modalityID else args.max_video_fps
            max_length = max_fps * (self.args.seconds_per_window * self.args.n_temporal_windows) # it should be only one window

            # TODO: Using nn.Embedding layers ??
            self.positional_embeddings[modalityID] = torch.nn.Parameter(
                torch.randn(1, max_length, self.args.model_args.latent_dim)
            )

            self.modality_embeddings[modalityID] = torch.nn.Parameter(
                torch.randn(1, max_length, self.args.model_args.latent_dim)
            )
        # -- -- special left- and right-hand modality embeddings   
        self.left_hand_embedding = torch.nn.Parameter(
            torch.randn(1, self.args.num_landmarks_per_hand, self.args.model_args.latent_dim)
        )
        self.right_hand_embedding = torch.nn.Parameter(
            torch.randn(1, self.args.num_landmarks_per_hand, self.args.model_args.latent_dim)
        )

        # -- transformer block
        self.transformer_block = torch.nn.Sequential(
            *[TransformerLayer(self.args) for _ in range(self.args.model_args.num_layers)],
        )

        # -- classification layer
        self.classification_layer = MultiHead(args)

    def forward(self, batch, mask=None):
        # -- processing the different modalities
        all_modalities = []
        for modalityID in self.args.modalities.keys():
            z = batch[f"modality:{modalityID}"]
            z = rearrange(z, "b ... d -> b (...) d")

            # -- -- projecting input data
            z = self.projections[modalityID](z)

            # -- -- adding positional embedding
            z = z + self.positional_embeddings[modalityID]

            # -- -- adding modality embedding
            z = z + self.modality_embeddings[modalityID]

            # -- -- adding special left- and right-hand embeddings
            if modalityID == "hand_landmarks":
                z[:, :self.num_landmarks_per_hand, :] = z[:, :self.num_landmarks_per_hand, :] + self.left_hand_embedding
                z[:, self.num_landmarks_per_hand:, :] = z[:, self.num_landmarks_per_hand:, :] + self.right_hand_embedding

            all_modalities.append(z)

        # -- concatenating all modalities
        x = torch.cat(all_modalities, dim=1)

        # -- -- applying transformer encoder
        x = self.transformer_block(x)

        if self.args.model_args.extracting_embeddings:
            return x

        # -- averaging embedding and applying classification
        output = ModelOutput(representation=x.mean(axis=1))
        return self.classification_layer(output)
