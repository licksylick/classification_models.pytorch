import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import make_weights_for_balanced_classes

class ClassificationDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.file_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloader(file_paths, labels, batch_size, transform, num_workers):
    dataset = ClassificationDataset(file_paths, labels, transform)

    weights = make_weights_for_balanced_classes(dataset.labels)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return dataloader
