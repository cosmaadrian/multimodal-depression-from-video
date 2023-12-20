import torch
from .repeat import repeat as multi_sequential_repeat
import math
import constants

from .perceiver_blocks import TransformerLayer

def fractional_positional_encoding(batch_size, d_model, length, downscale_factor):
    pe = torch.zeros(batch_size, length, d_model).to(downscale_factor.device)

    position = torch.arange(0, length).unsqueeze(1).tile((batch_size, )).to(downscale_factor.device)
    position = position * (1 / downscale_factor)
    position = position.T.unsqueeze(-1)

    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *-(math.log(10000.0) / d_model)))
    div_term = div_term.to(downscale_factor.device)

    sin_positions = torch.sin(position * div_term)
    cos_positions = torch.cos(position * div_term)

    pe[:, :, 0::2] = sin_positions
    pe[:, :, 1::2] = cos_positions

    return pe

class NoOpEncoder(torch.nn.Module):
    def __init__(self, args, modality_encoder_args):
        super(NoOpEncoder, self).__init__()
        self.args = args
        self.modality_encoder_args = modality_encoder_args

        self.batch_norm  = torch.nn.BatchNorm1d(
            self.modality_encoder_args.input_dim,
        )

        self.projection = torch.nn.Linear(
            self.modality_encoder_args.input_dim,
            self.modality_encoder_args.model_args.latent_dim,
            bias = False,
        )

        self.is_audio = "audio" in self.modality_encoder_args.name

        if 'original' in self.args.dataset:
            max_fps = 1
        else:
            max_fps = constants.MAX_AUDIO_FPS if self.is_audio else constants.MAX_VIDEO_FPS

        self.max_data_length = max_fps * self.args.seconds_per_window

    def forward(self, data, mask, framerate_ratio):
        data = data.view(data.shape[0], -1, self.modality_encoder_args.input_dim)
        data = self.batch_norm(data.permute(0, 2, 1)).permute(0, 2, 1)
        data = self.projection(data)

        downscale_factor = torch.ones((data.shape[0], )).float().to(data.device) if self.is_audio else framerate_ratio
        pe = fractional_positional_encoding(batch_size = data.shape[0], d_model = self.modality_encoder_args.model_args.latent_dim, length = self.max_data_length, downscale_factor = downscale_factor)
        pe = pe.to(data.device)
        data = data + pe

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
            bias = False
        )

        self.hand_type_embeddings = torch.nn.Embedding(
            2, self.modality_encoder_args.model_args.latent_dim,
        )

        self.max_data_length = constants.MAX_VIDEO_FPS * self.args.seconds_per_window

        self.encoder = multi_sequential_repeat(
            self.modality_encoder_args.model_args.num_layers,
            lambda _: TransformerLayer(
                modality_encoder_args,
            ),
            self.modality_encoder_args.model_args.layer_dropout_rate,
        )

    def forward(self, data, mask, framerate_ratio):
        batch, time, _, _, _ = data.shape

        data = data.view(batch, time * 2, -1).permute(0, 2, 1)
        data = self.batch_norm(data).permute(0, 2, 1)

        data = data.view(batch, time, 2, -1)

        downscale_factor = framerate_ratio

        data = data.view(data.shape[0], data.shape[1], 2, -1)
        data = self.projection(data)

        pe = fractional_positional_encoding(batch_size = data.shape[0], d_model = self.modality_encoder_args.model_args.latent_dim, length = self.max_data_length, downscale_factor = downscale_factor)

        pe = pe.repeat_interleave(2, dim = 1)
        pe = pe.reshape(-1, self.max_data_length, 2, self.modality_encoder_args.model_args.latent_dim)
        pe = pe.to(data.device)

        data = data + pe

        hand_type_ids = torch.tensor([0, 1]).repeat(self.max_data_length)
        hand_type_ids = hand_type_ids.to(data.device)

        hand_embeddings = self.hand_type_embeddings(hand_type_ids)
        hand_embeddings = hand_embeddings.reshape(self.max_data_length, 2, self.modality_encoder_args.model_args.latent_dim)

        data = data + hand_embeddings

        data = data.view(data.shape[0], -1, self.modality_encoder_args.model_args.latent_dim)

        mask = mask.repeat_interleave(2, dim = 1)

        data, _ = self.encoder(data, mask)

        data = data.view(data.shape[0], -1, 2, self.modality_encoder_args.model_args.latent_dim)
        data = data.mean(dim = 2)

        return data

class LandmarkEncoder(torch.nn.Module):
    def __init__(self, args, modality_encoder_args):
        super(LandmarkEncoder, self).__init__()
        self.args = args
        self.modality_encoder_args = modality_encoder_args

        self.batch_norm = torch.nn.BatchNorm1d(
            self.modality_encoder_args.input_dim,
        )

        self.projection = torch.nn.Linear(
            self.modality_encoder_args.input_dim,
            self.modality_encoder_args.model_args.latent_dim,
            bias = False,
        )

        if 'original' in self.args.dataset:
            self.max_data_length = self.args.seconds_per_window
        else:
            self.max_data_length = constants.MAX_VIDEO_FPS * self.args.seconds_per_window

        self.encoder = multi_sequential_repeat(
            self.modality_encoder_args.model_args.num_layers,
            lambda _: TransformerLayer(
                modality_encoder_args,
            ),
            self.modality_encoder_args.model_args.layer_dropout_rate,
        )

    def forward(self, data, mask, framerate_ratio):
        downscale_factor = framerate_ratio

        data = data.view(data.shape[0], data.shape[1], -1).permute(0, 2, 1)
        data = self.batch_norm(data).permute(0, 2, 1)

        data = self.projection(data)

        pe = fractional_positional_encoding(batch_size = data.shape[0], d_model = self.modality_encoder_args.model_args.latent_dim, length = self.max_data_length, downscale_factor = downscale_factor)
        pe = pe.to(data.device)
        data = data + pe
        data, _ = self.encoder(data, mask)

        return data

class BlinkingEncoder(torch.nn.Module):
    def __init__(self, args, modality_encoder_args):
        super(BlinkingEncoder, self).__init__()
        self.args = args
        self.modality_encoder_args = modality_encoder_args

        self.blinking_embeddings = torch.nn.Embedding(
            2, self.modality_encoder_args.model_args.latent_dim,
        )

        self.max_data_length = constants.MAX_VIDEO_FPS * self.args.seconds_per_window

    def forward(self, data, mask, framerate_ratio):
        downscale_factor = framerate_ratio

        data = self.blinking_embeddings(data.type(torch.int32))

        pe = fractional_positional_encoding(batch_size = data.shape[0], d_model = self.modality_encoder_args.model_args.latent_dim, length = self.max_data_length, downscale_factor = downscale_factor)
        pe = pe.to(data.device)
        data = data + pe

        return data


