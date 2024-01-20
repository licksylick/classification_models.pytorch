import os
import numpy as np
import pandas as pd
import importlib
import torch
from torchvision import transforms
from omegaconf import OmegaConf


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


def save_config(config):
    os.makedirs(config['common']['save_path'], exist_ok=True)
    with open(f"{config['common']['save_path']}/config.yaml", 'w') as file:
        OmegaConf.save(config=config, f=file)


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        convert_tensor = transforms.ToTensor()
        data = convert_tensor(data)
        data = data.unsqueeze(0)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def make_weights_for_balanced_classes(labels):
    num_classes = np.unique(labels)

    class_counts = [0] * num_classes

    for label in labels:
        class_counts[int(label)] += 1

        # Calculate weights for each class
    weights = [1.0 / class_counts[int(label)] if class_counts[int(label)] > 0 else 0.0 for label in labels]

    return weights