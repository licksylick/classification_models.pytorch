import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transforms import transform
from typing import Optional
from config import IMAGE_SIZE, BATCH_SIZE, VAL_SIZE, NUM_WORKERS, SEED, DATA_PATH


class VehicleData(Dataset):

    def __init__(self, csv_file, train=True, transform=None):
        super().__init__()

        if train:
            self.csv_file = csv_file
            df = pd.read_csv(self.csv_file)
            self.data_part_path = os.path.join(DATA_PATH, 'Images')
        else:
            self.csv_file = os.path.join('test.csv')
            df = pd.read_csv(self.csv_file)
            self.data_part_path = os.path.join(DATA_PATH, 'Images')
        unique_labels = torch.tensor(df['class'].values)
        self.idx_to_labels = zip(unique_labels, np.arange(len(unique_labels)))

        self.data = list()
        self.targets = list()
        for path in df.drop('class', axis=1).values:
            for img_path in path:
                self.data.append(os.path.join(self.data_part_path, img_path))
        for label in df.drop('filename', axis=1).values:
            for one_lbl in label:
                self.targets.append(one_lbl)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


class VehicleDataModule(pl.LightningDataModule):

    def __init__(
            self, data_path, image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE, val_size=VAL_SIZE, num_workers=NUM_WORKERS
    ):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.transforms = transform()

    def setup(self, stage: Optional[str] = None):
        self.train_data = VehicleData(self.data_path, transform=self.transforms)
        self.test_data = VehicleData(self.data_path, train=False, transform=self.transforms)

        indices = np.arange(len(self.train_data))
        np.random.seed(SEED)
        np.random.shuffle(indices)
        val_len = int(len(indices) * self.val_size)
        train_indices, val_indices = indices[val_len:], indices[:val_len]

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
