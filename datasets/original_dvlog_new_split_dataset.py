import torch
import random
import copy
import math
import numpy as np
import pandas as pd

import lib
from lib.dataset_extra import AcumenDataset

class OriginalDVlogNewSplitDataset(AcumenDataset):
    def __init__(self, args, kind = 'train', data_transforms = None):
        super().__init__(args = args, kind = kind, data_transforms = data_transforms)

        from lib import nomenclature
        self.nomenclature = nomenclature

        self.df = pd.read_csv(f'{args.environment["d-vlog"]}/splits/original/splits.csv')
        self.df['duration'] = self.df['duration'].apply(lambda x: math.floor(float(x.replace(',', '.'))))

        if kind in ['train']:
            not_missed_video_ids = pd.read_csv(f'{args.environment["d-vlog"]}/splits/training.csv')['video_id'].tolist()

        if kind in ['validation']:
            not_missed_video_ids = pd.read_csv(f'{args.environment["d-vlog"]}/splits/validation.csv')['video_id'].tolist()

        if kind in ['test']:
            not_missed_video_ids = pd.read_csv(f'{args.environment["d-vlog"]}/splits/test.csv')['video_id'].tolist()

        self.df = self.df[self.df['video_id'].isin(not_missed_video_ids)]
        self.df['video_id'] = self.df.index.astype(str)

        if kind in ['validation', 'test']:
            self.df = self.df.sort_values(by = 'duration')
            self.args.n_temporal_windows = 1

        self.df['label'] = self.df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        self.df['gender'] = self.df['gender'].apply(lambda x: 0 if x == 'f' else 1)

        self.modalities = {
            modality.name: self.nomenclature.MODALITIES[modality.name](df = self.df, env_path = f'{args.environment["d-vlog-original"]}', args = self.args)
            for modality in self.args.modalities
        }

        self.presence_masks = {
            video_sample["video_id"]: np.ones(video_sample["duration"]).astype(bool)
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

    def get_random_window(self, video_sample):
        presence_mask = self.presence_masks[video_sample["video_id"]]

        try:
            start_index = np.random.choice(np.argwhere(presence_mask).squeeze(-1), 1)[0]
        except ValueError as e:
            start_index = np.random.choice(np.argwhere(presence_mask == 0).squeeze(-1), 1)[0]

        start_in_seconds = start_index
        end_in_seconds = start_in_seconds + self.args.seconds_per_window

        if end_in_seconds > video_sample["duration"]:
            end_in_seconds = video_sample["duration"]
            start_in_seconds = end_in_seconds - self.args.seconds_per_window

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
        output['audio_frame_rate'] = 1.0
        output['video_frame_rate'] = 1.0
        output['start_in_seconds'] = start_in_seconds
        output['end_in_seconds'] = end_in_seconds

        return output

class OriginalDVlogNewSplitEvaluationDataset(OriginalDVlogNewSplitDataset):
    def __init__(self, args, kind='validation', data_transforms=None):

        local_args = copy.deepcopy(args)
        super().__init__(local_args, kind, data_transforms)

    def get_next_window(self, video_sample, window_offset = 0):
        video_frame_rate =  1.0
        presence_mask = self.presence_masks[video_sample["video_id"]]

        video_template = np.ones(presence_mask.shape)
        start_index = np.argwhere(video_template).ravel()[window_offset]
        end_index = int(start_index + self.args.seconds_per_window * video_frame_rate)
        last_window_idx = np.argwhere(video_template).ravel()[-1]

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

        start_in_seconds = int(start_index / video_frame_rate)
        end_in_seconds = int(start_in_seconds + self.args.seconds_per_window)

        difference = next_window_offset - window_offset
        satisfy_presence_thr = True if presence_mask[start_index] else False

        return start_in_seconds, end_in_seconds, is_last, next_window_offset, difference, last_window_idx, satisfy_presence_thr

    def get_batch(self, video_id, window_offset):
        video_sample = self.df[self.df['video_id'] == video_id].iloc[0]
        start_in_seconds, end_in_seconds, is_last, next_window_offset, difference, last_window_idx, satisfy_presence_thr = self.get_next_window(video_sample, window_offset = window_offset)

        output = {}
        for modality in self.args.modalities:
            chunk, no_modality_mask = self.modalities[modality.name].read_chunk(video_sample, start_in_seconds, end_in_seconds)
            chunk, mask = self.modalities[modality.name].post_process(chunk, no_modality_mask)
            output[f'modality:{modality.name}:data'] = chunk
            output[f'modality:{modality.name}:mask'] = mask

        output['labels'] = video_sample['label']
        output['gender'] = video_sample['gender']
        output['video_id'] = video_sample['video_id']
        output['audio_frame_rate'] = 1.0
        output['video_frame_rate'] = 1.0
        output['start_in_seconds'] = start_in_seconds
        output['end_in_seconds'] = end_in_seconds
        output['is_last'] = is_last
        output['satisfy_presence_thr'] = satisfy_presence_thr
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
