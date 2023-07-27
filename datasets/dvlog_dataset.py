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

        # computing face and voice presence for each video sample
        self.face_and_voice_presences = {}
        for index, video_sample in self.df.iterrows():
            self.face_and_voice_presences[video_sample["video_id"]] = self.compute_face_and_voice_presence(video_sample)

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

    def compute_face_and_voice_presence(self, video_sample):
        # loading voice and face frame-wise mask
        voice_mask = self.modalities["audio_embeddings"].masks[video_sample["video_id"]]
        face_mask = self.modalities["face_embeddings"].masks[video_sample["video_id"]]

        # computing the frame length of all windows according to the corresponding frame rate
        window_voice_frame_length = int(self.window_second_length * video_sample["audio_frame_rate"])
        window_face_frame_length = int(self.window_second_length * video_sample["video_frame_rate"])

        # convolving the mask sequence to compute the percentage of face and voice presence in each window set
        face_kernel = np.ones((window_face_frame_length,)) * (1 / window_face_frame_length)
        face_window_presence = np.convolve(face_mask, face_kernel, mode="same")  > self.args.presence_threshold

        voice_kernel = np.ones((window_voice_frame_length,)) * (1 / window_voice_frame_length)
        voice_window_convolved = np.convolve(voice_mask, voice_kernel, mode="same")
        
        # aligning voice w.r.t. the video frame rate
        aligning_idxs = np.linspace(0, voice_window_convolved.shape[0], num = face_window_presence.shape[0]).astype(np.int32)
        aligning_idxs[-1] -= 1 # implementation detail
        aligned_voice_window_convolved = voice_window_convolved[aligning_idxs]
        aligned_voice_window_presence = aligned_voice_window_convolved > self.args.presence_threshold

        # combining face and voice presence
        face_and_voice_presence = np.logical_and(face_window_presence, aligned_voice_window_presence)

        return face_and_voice_presence

    def get_random_window(self, video_sample):
        face_and_voice_presence = self.face_and_voice_presences[video_sample["video_id"]]

        # finding a random window where both face and voice are present
        start_index = np.random.choice(np.argwhere(face_and_voice_presence).squeeze(-1), 1)[0]

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
