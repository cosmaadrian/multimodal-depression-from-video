import torch
import pandas as pd

import lib
from lib.dataset_extra import AcumenDataset


class DVlogDataset(AcumenDataset):
    def __init__(self, args, kind = 'train', data_transforms = None):
        super().__init__(args = args, kind = kind, data_transforms = data_transforms)

        from lib import nomenclature
        self.nomenclature = nomenclature

        if kind == 'train':
            self.df = pd.read_csv('./data/databases/D-vlog/splits/training.csv')

        if kind == 'validation':
            self.df = pd.read_csv('./data/databases/D-vlog/splits/validation.csv')

        if kind == 'test':
            self.df = pd.read_csv('./data/databases/D-vlog/splits/test.csv')

        self.modalities = {
            modality: self.nomenclature.MODALITY[modality](args = self.args)
            for modality in self.args.modalities
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
            num_workers = 4,
            pin_memory = True,
        )

    def get_random_window(self, video):
        # TODO get a random window ?
        return video['video_id'] + '.npz'

    def __getitem__(self, idx):
        video = self.df.iloc[idx]
        window = self.get_random_window(video)

        output = {}
        for modality in self.args.modalities:
            output[modality] = self.modalities[modality].read_chunk(window)

        return output
