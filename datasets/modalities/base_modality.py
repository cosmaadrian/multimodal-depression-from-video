import sys
import glob
import numpy as np

class Modality(object):
    def __init__(self, args):
        self.args = args
        self.chunk_cache = {}

    def _get_no_feats_idxs_list_(self, video_id):
        if "face" in self.modality_dir:
            filepath = "no_face_idxs.npz"
        elif "body" in self.modality_dir:
            filepath = "no_body_idxs.npz"
        elif "hand" in self.modality_dir:
            filepath = "no_hand_idxs.npz"
        elif "audio" in self.modality_dir:
            filepath = "voice_activity.npz"
        else:
            raise ValueError(f"Modality no identified from directory: {self.modality_dir}")
        
        return np.load(f'{self.args.environment["d-vlog"]}/data/{video_id}/{filepath}')['data']

    def _indexes_from_chunkfiles_(self, video_id):
        if video_id in self.chunk_cache:
            return self.chunk_cache[video_id]

        chunk_files = glob.glob(f'{self.args.environment["d-vlog"]}/data/{video_id}/{self.modality_dir}/*.npz')
        indexes = [(int(chunk_file.split('/')[-1].split('.')[0].split('_')[-2]), int(chunk_file.split('/')[-1].split('.')[0].split('_')[-1])) for chunk_file in chunk_files]
        self.chunk_cache[video_id] = indexes

        return indexes

    def read_chunk(self, video, start, end):
        video_id = video["video_id"]
        fps =  100 if "audio" in self.modality_dir else int(video["video_frame_rate"])

        # finding out left and right bounds
        start_frame = int(start * fps)
        end_frame = int(end * fps)

        indexes = self._indexes_from_chunkfiles_(video_id)
        
        min_index = min([v for v in indexes if v[0] <= start_frame], key = lambda x: abs(x[0] - start_frame))[0]
        max_index = min([v for v in indexes if v[1] >= end_frame], key = lambda x: abs(x[1] - end_frame))[1]

        # which chunk files are needed?
        files_in_window = sorted([v for v in indexes if v[0] >= min_index and v[1] <= max_index])

        # loading no-feature idxs list
        all_no_feats_idxs = self._get_no_feats_idxs_list_(video_id)

        # building temporal window
        previous_len = 0
        chunk_no_feats_idxs = []
        for i, (start, end) in enumerate(files_in_window):
            data = np.load(f'{self.args.environment["d-vlog"]}/data/{video_id}/{self.modality_dir}/{video_id}_{str(start).zfill(6)}_{str(end).zfill(6)}.npz')['data']

            start_index = 0
            end_index = data.shape[0]

            if start_frame > start:
                start_index = start_frame - start

            if end_frame < end:
                end_index = end_frame - start

            if i == 0:
                output = data[start_index:end_index]
            else:
                output = np.concatenate((output, data[start_index:end_index]))

            # identifying no-feature frames
            for i, idx in enumerate(range(start_index, end_index)):
                if "audio" not in self.modality_dir:
                   if (start + idx) in all_no_feats_idxs:
                        relative_chunk_idx = previous_len + i
                        chunk_no_feats_idxs.append(relative_chunk_idx)
                else:
                    is_voice = False
                    for vad_slot in all_no_feats_idxs:
                        current_time = (start + idx) / fps
                        if current_time >= vad_slot[0] and current_time <= vad_slot[1]:
                            is_voice = True
                    if not is_voice:
                        relative_chunk_idx = previous_len + i
                        chunk_no_feats_idxs.append(relative_chunk_idx)

            previous_len = len(chunk_no_feats_idxs)

        # splitting windows
        output = np.asarray(np.split(output, self.args.n_temporal_windows, axis=0) )
        
        # applying normalization
        if 'embeddings' not in self.modality_dir:
            output = (output - np.mean(output, axis=0)) / np.std(output, axis=0)

        # flattening last dimensions -> (windows, frames, embed_size)
        output = np.reshape(output, (output.shape[0], output.shape[1], -1))

        return output.astype('float32'), chunk_no_feats_idxs

    def post_process(self, data, chunk_no_feats_idxs):
        """Post-processing input data.

        Args:
            data (np.array): input data (num_windows, num_frames, embed_size)
            chunk_no_feats_idxs: list of frame indeces where features where not found

        Returns:
            np.array: normalized and padded input data (num_windows, max_num_frames, embed_size)
            np.array: mask (num_windows, max_num_frames)
        """
        W, N, D = data.shape

        # padding to the max length
        max_fps = self.args.max_audio_fps if "audio" in self.modality_dir else self.args.max_video_fps
        frame_max_length = int(self.args.seconds_per_window) * int(max_fps)
        dif_with_max = frame_max_length - N

        pad_data = np.pad(data, [(0,0), (0, dif_with_max), (0,0)], mode="constant", constant_values=0)

        # computing mask
        # TODO check if mask is correct (1 = masked, 0 = not masked ???)
        mask = np.ones((W, N))
        mask = np.pad(mask, [(0,0), (0, dif_with_max)], mode="constant", constant_values=0)

        # identifying frames where no features where detected
        for idx in chunk_no_feats_idxs:
            window_idx = idx // mask.shape[1]
            frame_idx = idx % mask.shape[1]
            mask[window_idx][frame_idx] = 0

        return pad_data, mask.astype(bool)
