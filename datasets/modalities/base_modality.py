import os
import glob
import math
import numpy as np
import constants

class Modality(object):
    def __init__(self, args):
        self.args = args
        self.chunk_cache = {}

        self.modality_masks = {
            video_sample["video_id"]: self._get_modality_mask(video_sample)
            for idx, video_sample in self.df.iterrows()
        }

        self.modality_presence_masks = {
            video_sample["video_id"]: self._compute_modality_presence_mask(video_sample)
            for idx, video_sample in self.df.iterrows()
        }

    def _get_modality_mask(self, video_sample):
        all_chunk_files = sorted(os.listdir(f'{self.env_path}/data/{video_sample["video_id"]}/{self.modality_dir}/'))
        video_frame_length = int(all_chunk_files[-1].split(".")[0].split("_")[-1])

        no_modality_idxs = np.load(f'{self.env_path}/data/{video_sample["video_id"]}/{self.modality_mask_file}')['data'].tolist()

        mask = np.ones((video_frame_length,))
        mask[no_modality_idxs] = 0.

        return mask

    def _compute_modality_presence_mask(self, video_sample):
        modality_mask = self.modality_masks[video_sample["video_id"]]

        frames_per_second = video_sample["audio_frame_rate"] if "audio" in self.modality_dir else video_sample["video_frame_rate"]
        window_frame_length = int(self.args.seconds_per_window * frames_per_second)

        conv_kernel = np.ones((window_frame_length,)) * (1 / window_frame_length)
        conv_mask = np.convolve(modality_mask, conv_kernel, mode="same")

        if "audio" in self.modality_dir:
            all_chunk_files = sorted(os.listdir(f'{self.env_path}/data/{video_sample["video_id"]}/{self.video_ref_modality}/'))
            video_frame_length_reference = int(all_chunk_files[-1].split(".")[0].split("_")[-1])

            aligning_idxs = np.linspace(0, conv_mask.shape[0], num = video_frame_length_reference).astype(np.int32)
            aligning_idxs[-1] -= 1
            aligned_conv_mask = conv_mask[aligning_idxs]

            presence_mask = aligned_conv_mask > self.args.presence_threshold

        else:
            presence_mask = conv_mask > self.args.presence_threshold

        return presence_mask

    def _indexes_from_chunkfiles_(self, video_id):
        if video_id in self.chunk_cache:
            return self.chunk_cache[video_id]

        chunk_files = glob.glob(f'{self.env_path}/data/{video_id}/{self.modality_dir}/*.npz')
        indexes = [(int(chunk_file.split('/')[-1].split('.')[0].split('_')[-2]), int(chunk_file.split('/')[-1].split('.')[0].split('_')[-1])) for chunk_file in chunk_files]
        self.chunk_cache[video_id] = indexes

        return indexes

    def read_chunk(self, video_sample, start_in_seconds, end_in_seconds):
        video_id = video_sample["video_id"]
        fps =  video_sample["audio_frame_rate"] if "audio" in self.modality_dir else video_sample["video_frame_rate"]

        indexes = self._indexes_from_chunkfiles_(video_id)
        indexes = sorted(indexes, key = lambda x: x[0])

        start_frame = int(start_in_seconds * fps)
        end_frame = min(
            int(end_in_seconds * fps),
            max(indexes, key = lambda x: x[1])[1]
        )

        start_frame = (start_frame // self.args.n_temporal_windows) * self.args.n_temporal_windows
        end_frame = (end_frame // self.args.n_temporal_windows) * self.args.n_temporal_windows

        min_index = min([v for v in indexes if v[0] <= start_frame], key = lambda x: abs(x[0] - start_frame))[0]
        max_index = min([v for v in indexes if v[1] >= end_frame], key = lambda x: abs(x[1] - end_frame))[1]

        files_in_window = sorted([v for v in indexes if v[0] >= min_index and v[1] <= max_index])

        for i, (start_chunk, end_chunk) in enumerate(files_in_window):
            data = np.load(f'{self.env_path}/data/{video_id}/{self.modality_dir}/{video_id}_{str(start_chunk).zfill(6)}_{str(end_chunk).zfill(6)}.npz')['data']

            start_index = 0
            end_index = data.shape[0]

            if start_frame > start_chunk:
                start_index = start_frame - start_chunk

            if end_frame < end_chunk:
                end_index = end_frame - start_chunk

            if i == 0:
                output = data[start_index:end_index]
            else:
                output = np.concatenate((output, data[start_index:end_index]))

        no_modality_mask = self.modality_masks[video_id][start_frame:end_frame]

        padding = 0
        mask_padding = 0
        original_shape = output.shape
        if output.shape[0] % self.args.n_temporal_windows != 0 or output.shape[0] % self.args.seconds_per_window != 0:
            padding = int(self.args.n_temporal_windows * self.args.seconds_per_window * np.ceil(fps)) - output.shape[0]
            if padding < 0:
                output = output[:-padding, ...]
                padding = int(self.args.n_temporal_windows * self.args.seconds_per_window * np.ceil(fps)) - output.shape[0]
            assert padding >= 0, "padding is negative"

            mask_padding = int(self.args.n_temporal_windows * self.args.seconds_per_window * np.ceil(fps)) - no_modality_mask.shape[0]
            if mask_padding < 0:
                no_modality_mask = no_modality_mask[:-padding, ...]
                mask_padding = int(self.args.n_temporal_windows * self.args.seconds_per_window * np.ceil(fps)) - no_modality_mask.shape[0]
            mask_padding = int(mask_padding)

            if (output.shape[0] + padding) % self.args.n_temporal_windows != 0 or (output.shape[0] + padding) % self.args.seconds_per_window != 0:
                padding = padding + 1
                mask_padding = mask_padding + 1

            output = np.concatenate((output, np.zeros((padding, *output.shape[1:]))))
            no_modality_mask = np.concatenate((no_modality_mask, np.zeros((mask_padding,)))).astype(bool)

        assert output.shape[0] % self.args.n_temporal_windows == 0, f"output shape {output.shape} is not divisible by n_temporal_windows ({self.args.n_temporal_windows}), padding = {padding}, {output.shape, original_shape, self.args.n_temporal_windows, self.args.seconds_per_window, fps, self.args.n_temporal_windows * self.args.seconds_per_window * fps}"
        assert output.shape[0] % self.args.seconds_per_window == 0, f"output shape {output.shape} is not divisible by seconds_per_window ({self.args.seconds_per_window}), padding = {padding}, {output.shape, original_shape, self.args.n_temporal_windows, self.args.seconds_per_window, fps, self.args.n_temporal_windows * self.args.seconds_per_window * fps}"

        output = np.asarray(np.split(output, self.args.n_temporal_windows, axis=0))
        no_modality_mask = np.asarray(np.split(no_modality_mask, self.args.n_temporal_windows, axis=0))

        assert output.shape[1] != 0, f'{video_id}:{self.modality_dir}: output shape ({original_shape} -> {output.shape}) is 0, {start_in_seconds}, {end_in_seconds}, {self.args.n_temporal_windows}, {self.args.seconds_per_window}, {video_sample["duration"]}'

        return output.astype('float32'), no_modality_mask

    def post_process(self, data, no_modality_mask):
        """Post-processing input data.

        Args:
            data (np.array): input data (num_windows, num_frames, embed_size)
            no_modality_mask (np.array): mask indicating in which frames modality was not found
        Returns:
            np.array: padded input data (num_windows, max_num_frames, embed_size)
            np.array: mask (num_windows, max_num_frames)
        """
        W, T = data.shape[0], data.shape[1]

        flatten_data = np.reshape(data, (W, T, -1))

        max_fps = constants.MAX_AUDIO_FPS if "audio" in self.modality_dir else constants.MAX_VIDEO_FPS
        frame_max_length = int(self.args.seconds_per_window * max_fps)
        dif_with_max = frame_max_length - T

        pad_data = np.pad(flatten_data, [(0,0), (0, dif_with_max), (0,0)], mode="constant", constant_values=0) if dif_with_max >= 0 else flatten_data[:, :frame_max_length, :]

        no_modality_mask = np.pad(no_modality_mask, [(0,0), (0, dif_with_max)], mode="constant", constant_values=0) if dif_with_max >= 0 else no_modality_mask[:, :frame_max_length]

        padding_mask = np.ones((W, T))
        padding_mask = np.pad(padding_mask, [(0,0), (0, dif_with_max)], mode="constant", constant_values=0) if dif_with_max >= 0 else padding_mask[:, :frame_max_length]

        mask = np.logical_and(padding_mask, no_modality_mask)

        original_shape = (W, -1) + data.shape[2:]
        out_data = np.reshape(pad_data, original_shape)

        return out_data, mask.astype(bool)
