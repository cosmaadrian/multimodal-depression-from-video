import torch
from einops import rearrange, repeat
from .repeat import repeat as multi_sequential_repeat
from einops.layers.torch import Reduce

from .perceiver_blocks import TransformerLayer


class NoOpEncoder(torch.nn.Module):
    def __init__(self, args, modality_encoder_args):
        super(NoOpEncoder, self).__init__()
        self.args = args
        self.modality_encoder_args = modality_encoder_args

        self.projection = torch.nn.Linear(
            self.modality_encoder_args.input_dim,
            self.modality_encoder_args.model_args.latent_dim,
        )

        max_fps = self.args.max_audio_fps if "audio" in self.modality_encoder_args.name else self.args.max_video_fps
        self.max_data_length = max_fps * self.args.n_temporal_windows * self.args.seconds_per_window

        self.positional_embeddings = torch.nn.Embedding(
            self.max_data_length, self.modality_encoder_args.model_args.latent_dim,
        )

    def forward(self, data, mask):
        data = self.projection(data)

        time_steps = torch.arange(self.max_data_length).to(data.device)
        data = data + self.positional_embeddings(time_steps)

        return data

class HandLandmarkEncoder(torch.nn.Module):
    def __init__(self, args, modality_encoder_args):
        super(HandLandmarkEncoder, self).__init__()
        self.args = args
        self.modality_encoder_args = modality_encoder_args

        self.batch_norm  = torch.nn.BatchNorm1d(
            self.modality_encoder_args.input_dim,
        )

        self.projection = torch.nn.Linear(
            self.modality_encoder_args.input_dim,
            self.modality_encoder_args.model_args.latent_dim,
        )

        self.hand_type_embeddings = torch.nn.Embedding(
            2, self.modality_encoder_args.model_args.latent_dim,
        )


        self.max_data_length = self.args.max_video_fps * self.args.n_temporal_windows * self.args.seconds_per_window
        self.positional_embeddings = torch.nn.Embedding(
            self.max_data_length,
            self.modality_encoder_args.model_args.latent_dim,
        )

        self.encoder = multi_sequential_repeat(
            self.modality_encoder_args.model_args.num_layers,
            lambda _: TransformerLayer(
                modality_encoder_args,
            ),
            self.modality_encoder_args.model_args.layer_dropout_rate,
        )

    def forward(self, data, mask):
        """Forward pass.
        
        Args:
            data (torch.Tensor): input data (batch, time, hands, landmarks, coordinates)
            mask (torch.Tensor): mask tensor (batch, time)

        Returns:
            torch.Tensor: modality encoder output tensor (batch, time, hands, latent_dim)
        """
        batch, time, _, _, _ = data.shape

        data = data.view(batch, time * 2, -1).permute(0,2,1)
        data = self.batch_norm(data).permute(0,2,1)

        data = data.view(batch, time, 2, -1)
        data = self.projection(data)

        # add positional embeddings
        time_steps = torch.arange(self.max_data_length).to(data.device)
        time_steps = time_steps.repeat_interleave(2)
        positions = self.positional_embeddings(time_steps)
        positions = positions.reshape(self.max_data_length, 2, self.modality_encoder_args.model_args.latent_dim)

        data = data + positions

        # add token_types
        hand_type_ids = torch.tensor([0, 1]).repeat(self.max_data_length)
        hand_type_ids = hand_type_ids.to(data.device)

        hand_embeddings = self.hand_type_embeddings(hand_type_ids)
        hand_embeddings = hand_embeddings.reshape(self.max_data_length, 2, self.modality_encoder_args.model_args.latent_dim)

        data = data + hand_embeddings

        # process with encoder
        data = data.view(data.shape[0], -1, self.modality_encoder_args.model_args.latent_dim)

        mask = mask.repeat_interleave(2, dim = -1)
        data, _ = self.encoder(data, mask)

        # reshape again
        data = data.view(data.shape[0], -1, 2, self.modality_encoder_args.model_args.latent_dim)
        data = data.mean(dim = 2)

        return data

class LandmarkEncoder(torch.nn.Module):
    def __init__(self, args, modality_encoder_args):
        super(LandmarkEncoder, self).__init__()
        self.args = args
        self.modality_encoder_args = modality_encoder_args

        self.batch_norm  = torch.nn.BatchNorm1d(
            self.modality_encoder_args.input_dim,
        )

        self.projection = torch.nn.Linear(
            self.modality_encoder_args.input_dim,
            self.modality_encoder_args.model_args.latent_dim,
        )

        self.max_data_length = self.args.max_video_fps * self.args.n_temporal_windows * self.args.seconds_per_window
        self.positional_embeddings = torch.nn.Embedding(
            self.max_data_length,
            self.modality_encoder_args.model_args.latent_dim,
        )

        self.encoder = multi_sequential_repeat(
            self.modality_encoder_args.model_args.num_layers,
            lambda _: TransformerLayer(
                modality_encoder_args,
            ),
            self.modality_encoder_args.model_args.layer_dropout_rate,
        )

        # TODO add batch normalization?

    def forward(self, data, mask):
        """Forward pass.
        
        Args:
            data (torch.Tensor): input data (batch, time, landmarks, coordinates)
            mask (torch.Tensor): mask tensor (batch, time)

        Returns:
            torch.Tensor: modality encoder output tensor (batch, time, latent_dim)
        """
        data = data.view(data.shape[0], data.shape[1], -1).permute(0,2,1)
        data = self.batch_norm(data).permute(0,2,1)

        data = self.projection(data)

        time_steps = torch.arange(self.max_data_length).to(data.device)
        data = data + self.positional_embeddings(time_steps)
        data, _ = self.encoder(data, mask)

        return data
