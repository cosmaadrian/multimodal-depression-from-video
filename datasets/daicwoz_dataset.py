import torch
import random
import copy
import math
import numpy as np
import pandas as pd

import lib
from lib.dataset_extra import AcumenDataset

class DaicWozDataset(AcumenDataset):
    def __init__(self, args, kind = 'train', data_transforms = None):
        super().__init__(args = args, kind = kind, data_transforms = data_transforms)

        from lib import nomenclature
        self.nomenclature = nomenclature

        if kind == 'train':
            self.df = pd.read_csv(f'{args.environment["daic-woz"]}/splits/training.csv', index_col=0)

        if kind == 'validation':
            self.df = pd.read_csv(f'{args.environment["daic-woz"]}/splits/validation.csv', index_col=0)
            self.df = self.df.sort_values(by = 'duration')

        if kind == 'test':
            self.df = pd.read_csv(f'{args.environment["daic-woz"]}/splits/test.csv', index_col=0)
            self.df = self.df.sort_values(by = 'duration')
            self.args.n_temporal_windows = 1

        self.df['video_id'] = self.df["Participant_ID"]
        self.df['label'] = self.df["PHQ8_Binary"]
        self.df['gender'] = self.df['Gender']
        self.df['duration'] = self._compute_duration()
        self.df['video_frame_rate'] = [30.0] * len(self.df.index)
        self.df['audio_frame_rate'] = [100.0] * len(self.df.index)

        self.modalities = {
            modality.name: self.nomenclature.MODALITIES[modality.name](df = self.df, args = self.args)
            for modality in self.args.modalities
        }

        # computing presence masks for each video sample
        self.presence_masks = {
            video_sample["video_id"]: self._compute_presence_mask(video_sample)
            for idx, video_sample in self.df.iterrows()
        }

    def __len__(self):
        return len(self.df.index)

    @classmethod
    def train_dataloader(cls, args):
        dataset = cls(args = args, kind = 'train')

        return torch.utils.data.DataLoader(
            dataset,
            num_workers = args.environment['num_workers'],
            pin_memory = True,
            shuffle = True,
            drop_last = True,
            batch_size = args.batch_size
        )

    @classmethod
    def val_dataloader(cls, args, kind = 'validation'):
        dataset = cls(args = args, data_transforms = None, kind = kind)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = args.environment['num_workers'],
            pin_memory = True,
        )

    def _compute_duration(self):
        durations = []
        for video_id in self.df['video_id'].tolist():
            audio_covarep_chunks = sorted( os.listdir(f'{args.environment["daic-woz"]}/data/{video_id}/audio_covarep/') )
            nframes = int(audio_covarep_chunks[-1].split(".")[0].split("_")[-1])
            durations.append( float(nframes / 100.0) )

        return durations

    def _compute_presence_mask(self, video_sample):
        # obtaining modality presence mask of a specific video sample
        presence_mask = self.modalities["audio_covarep"].modality_presence_masks[video_sample["video_id"]]
        return presence_mask

    def get_random_window(self, video_sample):
        # obtaining presence mask of a specific video sample
        presence_mask = self.presence_masks[video_sample["video_id"]]

        # finding a random window where priority modalities are present
        try:
            start_index = np.random.choice(np.argwhere(presence_mask).squeeze(-1), 1)[0]
        except ValueError as e:
            start_index = np.random.choice(np.argwhere(presence_mask == 0).squeeze(-1), 1)[0]

        # computing window in seconds
        start_in_seconds = start_index // video_sample["video_frame_rate"]
        end_in_seconds = start_in_seconds + int(self.args.n_temporal_windows * self.args.seconds_per_window)

        if end_in_seconds > video_sample["duration"]:
            end_in_seconds = int(video_sample["duration"])
            start_in_seconds = end_in_seconds - int(self.args.n_temporal_windows * self.args.seconds_per_window)

        if start_in_seconds < 0:
            start_in_seconds = 0

        return start_in_seconds, end_in_seconds

    def __getitem__(self, idx):
        video_sample = self.df.iloc[idx]
        start_in_seconds, end_in_seconds = self.get_random_window(video_sample)

        output = {}
        for modality in self.args.modalities:
            chunk, no_modality_mask = self.modalities[modality.name].read_chunk(video_sample, start_in_seconds, end_in_seconds)
            chunk, mask = self.modalities[modality.name].post_process(chunk, no_modality_mask)
            output[f'modality:{modality.name}:data'] = chunk
            output[f'modality:{modality.name}:mask'] = mask

        output['labels'] = video_sample['label']
        output['gender'] = video_sample['gender']
        output['video_id'] = video_sample['video_id']
        output['audio_frame_rate'] = video_sample["audio_frame_rate"]
        output['video_frame_rate'] = video_sample["video_frame_rate"]
        output['start_in_seconds'] = start_in_seconds
        output['end_in_seconds'] = end_in_seconds

        return output

class DaicWozEvaluationDataset(DaicWozDataset):
    def __init__(self, args, kind='validation', data_transforms=None):

        local_args = copy.deepcopy(args)

        if kind in ['validation', 'test']:
            local_args.presence_threshold = -1

        super().__init__(local_args, kind, data_transforms)

    def get_next_window(self, video_sample, window_offset = 0):
        # obtaining presence mask of a specific video sample
        presence_mask = self.presence_masks[video_sample["video_id"]]

        # finding the first window where priority modalities are present
        start_index = np.argwhere(presence_mask).ravel()[window_offset]
        end_index = int(start_index + self.args.seconds_per_window * video_sample["video_frame_rate"])
        last_window_idx = np.argwhere(presence_mask).ravel()[-1]

        is_last = 0
        if window_offset >= last_window_idx:
            is_last = 1
            next_window_offset = window_offset
        else:
            try:
                next_window_offset = np.argwhere(((np.argwhere(presence_mask) - end_index) > 0)).ravel()[0]
            except:
                is_last = 1
                next_window_offset = window_offset

        # computing window in seconds
        start_in_seconds = start_index / video_sample["video_frame_rate"]
        end_in_seconds = start_in_seconds + self.args.seconds_per_window

        difference = next_window_offset - window_offset

        return start_in_seconds, end_in_seconds, is_last, next_window_offset, difference, last_window_idx

    def get_batch(self, video_id, window_offset):
        video_sample = self.df[self.df['video_id'] == video_id].iloc[0]
        start_in_seconds, end_in_seconds, is_last, next_window_offset, difference, last_window_idx = self.get_next_window(video_sample, window_offset = window_offset)

        output = {}
        for modality in self.args.modalities:
            chunk, no_modality_mask = self.modalities[modality.name].read_chunk(video_sample, start_in_seconds, end_in_seconds)
            chunk, mask = self.modalities[modality.name].post_process(chunk, no_modality_mask)
            output[f'modality:{modality.name}:data'] = chunk
            output[f'modality:{modality.name}:mask'] = mask

        output['labels'] = video_sample['label']
        output['gender'] = video_sample['gender']
        output['video_id'] = video_sample['video_id']
        output['audio_frame_rate'] = video_sample["audio_frame_rate"]
        output['video_frame_rate'] = video_sample["video_frame_rate"]
        output['start_in_seconds'] = start_in_seconds
        output['end_in_seconds'] = end_in_seconds
        output['is_last'] = is_last
        output['next_window_offset'] = next_window_offset
        output['total_windows'] = last_window_idx
        output['differences'] = difference

        return output

    def __getitem__(self, idx):
        video_sample = self.df.iloc[idx]

        output = {}
        output['video_id'] = video_sample['video_id']
        output['labels'] = video_sample['label']
        output['next_window_offset'] = 0
        output['total_windows'] = self.presence_masks[output['video_id']].sum()

        return output
