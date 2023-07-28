import torch
import random
import numpy as np
import pandas as pd

import lib
from lib.dataset_extra import AcumenDataset

class DVlogDataset(AcumenDataset):
    def __init__(self, args, kind = 'train', data_transforms = None):
        super().__init__(args = args, kind = kind, data_transforms = data_transforms)

        from lib import nomenclature
        self.nomenclature = nomenclature

        if kind == 'train':
            self.df = pd.read_csv(f'{args.environment["d-vlog"]}/splits/training.csv', index_col=0)

        if kind == 'validation':
            self.df = pd.read_csv(f'{args.environment["d-vlog"]}/splits/validation.csv', index_col=0)

        if kind == 'test':
            self.df = pd.read_csv(f'{args.environment["d-vlog"]}/splits/test.csv', index_col=0)

        self.df['label'] = self.df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        self.df['gender'] = self.df['gender'].apply(lambda x: 0 if x == 'f' else 1)

        self.modalities = {
            modality.name: self.nomenclature.MODALITIES[modality.name](df = self.df, args = self.args)
            for modality in self.args.modalities
        }

        self.window_second_length = int(self.args.n_temporal_windows * self.args.seconds_per_window)

        # identifying priority modalities
        self.priority_modalities = self._priority_modalities()

        # TODO Discuss this thing together
        # special cases where samples have to be removed, since the priority modality was not found
        if "body_landmarks" in self.priority_modalities:
            self.df = self.df.drop(self.df[self.df["body_presence"] < (self.df["duration"] * self.args.presence_threshold)].index)
        if "hand_landmarks" in self.priority_modalities:
            self.df = self.df.drop(self.df[self.df["hand_presence"] < (self.df["duration"] * self.args.presence_threshold)].index)

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

    def _priority_modalities(self):
        modality_names = [modality.name for modality in self.args.modalities]
        if len(self.args.modalities) == 1:
            # mono-modality experiments
            return [self.args.modalities[0].name]
        else:
            # multi-modality experiments
            # prefence: audio_embeddings > face_embeddings > face_landmarks > body_landmarks > hand_landmarks          
            if ("audio_embeddings" in modality_names) and ("face_embeddings" in modality_names):
                return ["audio_embeddings", "face_embeddings"]
            
            if ("audio_embeddings" in modality_names) and ("face_landmarks" in modality_names):
                return ["audio_embeddings", "face_landmarks"]
            
            if "audio_embeddings" in modality_names:
                return ["audio_embeddings"]
            
            if "face_embeddings" in modality_names:
                return ["face_embeddings"]
            
            if "face_landmarks" in modality_names:
                return ["face_landmarks"]
            
            if "body_landmarks" in modality_names:
                return ["body_landmarks"]
            
            raise Exception("Madonna! No modality was identified when computing the presence mask")
            
            # hand landmarks is the least frequent modality, so it will 
            # never be a priority one unless in mono-modality experiments

    def _compute_presence_mask(self, video_sample):
        # obtaining modality presence mask of a specific video sample
        presence_mask = self.modalities[self.priority_modalities[0]].modality_presence_masks[video_sample["video_id"]]
        # combining more than one modality mask: only in the case voice + face
        for other_modality_id in self.priority_modalities[1:]:
            other_presence_mask = self.modalities[other_modality_id].modality_presence_masks[video_sample["video_id"]]
            presence_mask = np.logical_and(presence_mask, other_presence_mask)

        return presence_mask

    def get_random_window(self, video_sample):
        # obtaining presence mask of a specific video sample   
        presence_mask = self.presence_masks[video_sample["video_id"]]

        # finding a random window where both face and voice are present
        start_index = np.random.choice(np.argwhere(presence_mask).squeeze(-1), 1)[0]

        # computing window in seconds
        start_in_seconds = start_index / video_sample["video_frame_rate"]
        end_in_seconds = start_in_seconds + self.window_second_length

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

        return output

# TODO DVlogDatasetEvaluation to get windows in order
