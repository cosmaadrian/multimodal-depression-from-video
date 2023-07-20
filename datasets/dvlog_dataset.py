import torch
import pandas as pd
import random

import lib
from lib.dataset_extra import AcumenDataset


class DVlogDataset(AcumenDataset):
    def __init__(self, args, kind = 'train', data_transforms = None):
        super().__init__(args = args, kind = kind, data_transforms = data_transforms)

        from lib import nomenclature
        self.nomenclature = nomenclature

        if kind == 'train':
            self.df = pd.read_csv(f'{args.environment["d-vlog"]}/splits/training.csv')

        if kind == 'validation':
            self.df = pd.read_csv(f'{args.environment["d-vlog"]}/splits/validation.csv')

        if kind == 'test':
            self.df = pd.read_csv(f'{args.environment["d-vlog"]}/splits/test.csv')

        self.df['label'] = self.df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        self.df['gender'] = self.df['gender'].apply(lambda x: 0 if x == 'f' else 1)

        self.modalities = {
            modality: self.nomenclature.MODALITIES[modality](args = self.args)
            for modality in self.args.modalities.keys()
        }

    def __len__(self):
        return len(self.df.index)

    @classmethod
    def train_dataloader(cls, args):
        dataset = cls(args = args, kind = 'train')

        return torch.utils.data.DataLoader(
            dataset,
            num_workers = 4,
            pin_memory = True,
            batch_size = args.batch_size
        )

    @classmethod
    def val_dataloader(cls, args, kind = 'validation'):
        dataset = cls(args = args, data_transforms = None, kind = kind)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = 1, # TODO change to 4
            pin_memory = True,
        )

    def get_random_window(self, video):
        video_length = float(video['duration'].replace(',', '.'))
        window_length = self.args.n_temporal_windows * self.args.seconds_per_window

        start = random.randint(0, int(video_length) - window_length)
        end = start + window_length

        return start, end

    def __getitem__(self, idx):
        video = self.df.iloc[idx]
        start, end = self.get_random_window(video)

        output = {}
        for modality in self.args.modalities.keys():
            chunk = self.modalities[modality].read_chunk(video, start, end).astype('float32')
            chunk, mask = self.modalities[modality].post_process(chunk)
            output['modality:' + modality] = (torch.from_numpy(chunk), torch.from_numpy(mask))
        
        # -- computing mask tensor
        
        output['labels'] = video['label']
        output['gender'] = video['gender']

        return output
