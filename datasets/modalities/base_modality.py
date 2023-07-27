import os
import glob
import numpy as np

class Modality(object):
    def __init__(self, args):
        self.args = args
        self.chunk_cache = {}

        # loading no-modality masks for each video sample
        self.masks = {}
        for video_id in self.df["video_id"].tolist():
            self.masks[video_id] = self._get_global_mask_(video_id)

    def _get_global_mask_(self, video_id):       
        # getting the total number of frames that compose the video
        all_chunk_files = sorted(os.listdir(f'{self.args.environment["d-vlog"]}/data/{video_id}/{self.modality_dir}/'))
        video_frame_length = int(all_chunk_files[-1].split(".")[0].split("_")[-1])

        # loading frames where no modality was detected
        no_modality_idxs = np.load(f'{self.args.environment["d-vlog"]}/data/{video_id}/{self.modality_mask_file}')['data'].tolist()

        # constructing mask
        mask = np.ones((video_frame_length,))
        mask[no_modality_idxs] = 0.

        return mask

    def _indexes_from_chunkfiles_(self, video_id):
        if video_id in self.chunk_cache:
            return self.chunk_cache[video_id]

        file_name = f'{self.args.environment["d-vlog"]}/data/{video_id}/{self.modality_dir}/*.npz'
        chunk_files = glob.glob(file_name)

        indexes = [(int(chunk_file.split('/')[-1].split('.')[0].split('_')[-2]), int(chunk_file.split('/')[-1].split('.')[0].split('_')[-1])) for chunk_file in chunk_files]
        self.chunk_cache[video_id] = indexes

        return indexes

    def read_chunk(self, video_sample, start_in_seconds, end_in_seconds):
        video_id = video_sample["video_id"]
        fps =  int(video_sample["audio_frame_rate"]) if "audio" in self.modality_dir else int(video_sample["video_frame_rate"])

        # finding out left and right bounds
        start_frame = int(start_in_seconds * fps)
        end_frame = int(end_in_seconds * fps)

        indexes = self._indexes_from_chunkfiles_(video_id)

        min_index = min([v for v in indexes if v[0] <= start_frame], key = lambda x: abs(x[0] - start_frame))[0]
        max_index = min([v for v in indexes if v[1] >= end_frame], key = lambda x: abs(x[1] - end_frame))[1]

        # which chunk files are needed?
        files_in_window = sorted([v for v in indexes if v[0] >= min_index and v[1] <= max_index])

        # building temporal window
        for i, (start_chunk, end_chunk) in enumerate(files_in_window):
            data = np.load(f'{self.args.environment["d-vlog"]}/data/{video_id}/{self.modality_dir}/{video_id}_{str(start_chunk).zfill(6)}_{str(end_chunk).zfill(6)}.npz')['data']

            start_index = 0
            end_index = data.shape[0]

            # special cases where in-window relative indeces are needed
            if start_frame > start_chunk:
                start_index = start_frame - start_chunk

            if end_frame < end_chunk:
                end_index = end_frame - start_chunk

            # special case where the window is split into different chunk files where we need to concatenate them
            if i == 0:
                output = data[start_index:end_index]
            else:
                output = np.concatenate((output, data[start_index:end_index]))

        # splitting into windows
        output = np.asarray( np.split(output, self.args.n_temporal_windows, axis=0) )

        # obtaining no-modality presence mask
        no_modality_mask = self.masks[video_id][start_frame:end_frame]
        no_modality_mask = np.asarray( np.split(no_modality_mask, self.args.n_temporal_windows, axis=0) )

        return output.astype('float32'), no_modality_mask

    def post_process(self, data, no_modality_mask):
        """Post-processing input data.

        Args:
            data (np.array): input data (num_windows, num_frames, embed_size)
            no_modality_mask (np.array): mask indicating in which frames modality was not found
        Returns:
            np.array: normalized and padded input data (num_windows, max_num_frames, embed_size)
            np.array: mask (num_windows, max_num_frames)
        """
        data_shape = data.shape
        W, T = data_shape[0], data_shape[1]

        # flatten data input array
        flatten_data = np.reshape(data, (W, T, -1))

        # padding to the max length
        max_fps = self.args.max_audio_fps if "audio" in self.modality_dir else self.args.max_video_fps
        frame_max_length = int(self.args.seconds_per_window * max_fps)
        dif_with_max = frame_max_length - T

        pad_data = np.pad(flatten_data, [(0,0), (0, dif_with_max), (0,0)], mode="constant", constant_values=0)

        # computing mask
        # TODO check if mask is correct (1 = masked, 0 = not masked ???)
        no_modality_mask = np.pad(no_modality_mask, [(0,0), (0, dif_with_max)], mode="constant", constant_values=0)

        padding_mask = np.ones((W, T))
        padding_mask = np.pad(padding_mask, [(0,0), (0, dif_with_max)], mode="constant", constant_values=0)

        mask = np.logical_and(padding_mask, no_modality_mask)

        # reshaping to the original shape
        original_shape = (W, -1) + data.shape[2:]
        out_data = np.reshape(pad_data, original_shape)

        # TODO? N = the max length of the temporal window

        return pad_data, mask.astype(bool)
