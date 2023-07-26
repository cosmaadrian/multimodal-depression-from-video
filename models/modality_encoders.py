import torch
from einops import rearrange, repeat
from einops.layers.torch import Reduce

from .perceiver_blocks import TransformerLayer

class NoOpEncoder(torch.nn.Module):
    def __init__(self, args, modality_encoder_args, modality_id):
        super(NoOpEncoder, self).__init__()
        self.args = args
        self.modality_encoder_args = modality_encoder_args
        self.modality_id  = modality_id

        self.projection = torch.nn.Linear(
            self.modality_encoder_args.input_size,
            self.modality_encoder_args.model_args.latent_dim,
        )

        max_fps = self.args.max_audio_fps if "audio" in self.modality_id else self.args.max_video_fps
        self.max_length = max_fps * self.args.seconds_per_window # it should be only one temporal window

        self.positional_embeddings = torch.nn.Embedding(
            self.max_length, self.modality_encoder_args.model_args.latent_dim,
        )

    def forward(self, data, mask):
        data = self.projection(data)

        time_steps = torch.arange(self.max_length).to(data.device)
        data = data + self.positional_embeddings(time_steps)

        return data

class HandLandmarkEncoder(torch.nn.Module):
    def __init__(self, args, modality_encoder_args, modality_id):
        super(HandLandmarkEncoder, self).__init__()
        self.args = args
        self.modality_encoder_args = modality_encoder_args
        self.modality_id = modality_id

        self.projection = torch.nn.Linear(
            self.modality_encoder_args.input_size,
            self.modality_encoder_args.model_args.latent_dim,
        )

        self.hand_embeddings = torch.nn.Embedding(
            2, self.modality_encoder_args.model_args.latent_dim,
        )

        self.positional_embeddings = torch.nn.Embedding(
            self.args.max_video_fps * self.args.seconds_per_window,
            self.modality_encoder_args.model_args.latent_dim,
        )

        self.encoder = repeat(
            self.modality_encoder_args.num_layers,
            lambda _: TransformerLayer(
                modality_encoder_args,
            ),
            self.modality_encoder_args.model_args.layer_dropout_rate,
        )

    def forward(self, data, mask):
        data = data.view(data.shape[0], data.shape[1], 2, -1)
        data = self.projection(data)

        # add positional embeddings
        time_steps = torch.arange(self.max_length).to(data.device)
        data = data + self.positional_embeddings(time_steps)

        # add token_types
        token_type_ids = torch.tensor([0, 1]).repeat(self.args.max_video_fps * self.args.seconds_per_window)
        token_type_ids = token_type_ids.to(data.device)

        token_embeddings = self.hand_embeddings(token_type_ids)

        data = data + token_embeddings

        # process with encoder
        data = data.view(data.shape[0], -1, self.modality_encoder_args.latent_dim)
        data = self.encoder(data)

        # reshape again
        data = data.view(data.shape[0], -1, 2, self.modality_encoder_args.latent_dim)
        data = data.mean(dim = 2)

        return data


class LandmarkEncoder(torch.nn.Module):
    def __init__(self, args, modality_encoder_args, modality_id):
        super(LandmarkEncoder, self).__init__()
        self.args = args
        self.modality_encoder_args = modality_encoder_args
        self.modality_id = modality_id

        self.projection = torch.nn.Linear(
            self.modality_encoder_args.input_size,
            self.modality_encoder_args.model_args.latent_dim,
        )

        self.positional_embeddings = torch.nn.Embedding(
            self.args.max_video_fps * self.args.seconds_per_window,
            self.modality_encoder_args.model_args.latent_dim,
        )

        self.encoder = repeat(
            self.modality_encoder_args.num_layers,
            lambda _: TransformerLayer(
                modality_encoder_args,
            ),
            self.modality_encoder_args.model_args.layer_dropout_rate,
        )

        # TODO add batch normalization?

    def forward(self, data, mask):
        data = data.view(data.shape[0], data.shape[1], -1)
        data = self.projection(data)

        time_steps = torch.arange(self.max_length).to(data.device)
        data = data + self.positional_embeddings(time_steps)
        data = self.encoder(data, mask = mask)
        return data
