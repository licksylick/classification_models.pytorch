import os
import warnings
import torch
import numpy as np
import pytorch_lightning as pl
import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold, train_test_split
from pytorch_lightning import seed_everything
from models.model import ClassificationModel
from transforms import Transforms
from utils import save_config
from data import create_dataloader

warnings.filterwarnings('ignore')


def preprocess_config(config):
    exp_name = config['common'].get('exp_name', 'exp0')
    config['common']['save_path'] = os.path.join(exp_name)

    # Overwrite some params
    max_epochs = config['common'].get('max_epochs', False)
    if max_epochs:
        config['trainer']['params']['max_epochs'] = max_epochs

        for opt_index, _ in enumerate(config['optimizers']):
            sch = config['optimizers'][opt_index].get('scheduler', False)
            if sch and 'LinearWarmupCosineAnnealingLR' in sch['target']:
                config['optimizers'][opt_index]['scheduler']['params']['max_epochs'] = max_epochs

    return config


def train_with_cv(config, model, train_val_file_paths, train_val_labels, test_dataloader,
                  train_transform, val_test_transform):
    log_dir = config['common']['exp_name']
    num_epochs = config['common']['max_epochs']
    batch_size = config['common']['batch_size']
    num_workers = config['common']['num_workers']
    n_splits = config['common']['num_splits']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_file_paths, train_val_labels)):
        print(f"Fold {fold + 1}/{skf.get_n_splits()}")

        train_files, val_files = np.array(train_val_file_paths)[train_idx].tolist(), np.array(train_val_file_paths)[
            val_idx].tolist()
        train_labels, val_labels = np.array(train_val_labels)[train_idx].tolist(), np.array(train_val_labels)[
            val_idx].tolist()

        train_dataloader = create_dataloader(train_files, train_labels, batch_size, train_transform,
                                             num_workers)
        val_dataloader = create_dataloader(val_files, val_labels, batch_size, val_test_transform,
                                           num_workers)

        trainer = pl.Trainer(default_root_dir=log_dir,
                             max_epochs=num_epochs)  # You can adjust gpus parameter based on your system
        trainer.callbacks += [getattr(importlib.import_module(callback_config.target.rsplit('.', 1)[0]),
                                      callback_config.target.rsplit('.', 1)[1])(**callback_config.params) for
                              callback_config in config.callbacks]
        trainer.fit(model, train_dataloader, val_dataloader)

    trainer.test(model, test_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cross_validation', type=str, default=False, help='Bool value (use cross-validation or no')
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()

    config = preprocess_config(OmegaConf.load(args.config))
    save_config(config)
    print(OmegaConf.to_yaml(config))

    seed_everything(config['common']['seed'], workers=True)

    train_val_file_paths = []  # List of file paths for your images
    train_val_labels = []  # List of corresponding labels (class indices)

    test_file_paths = []
    test_labels = []

    dataset_dir = config['dataset']['path']

    for stage_dir in os.listdir(dataset_dir):
        if stage_dir == 'test':
            for i, class_dir in enumerate(os.listdir(os.path.join(dataset_dir, stage_dir))):
                for file in os.listdir(os.path.join(dataset_dir, stage_dir, class_dir)):
                    test_file_paths.append(os.path.join(dataset_dir, stage_dir, class_dir, file))
                    test_labels.append(i)
        else:
            for i, class_dir in enumerate(os.listdir(os.path.join(dataset_dir, stage_dir))):
                for file in os.listdir(os.path.join(dataset_dir, stage_dir, class_dir)):
                    train_val_file_paths.append(os.path.join(dataset_dir, stage_dir, class_dir, file))
                    train_val_labels.append(i)

    model = ClassificationModel(config)

    transforms = Transforms(config['dataset']['img_size'])
    train_transform, val_test_transform = transforms.train_transform(), transforms.val_test_transform()
    num_workers = config['common']['num_workers']

    test_dataloader = create_dataloader(test_file_paths, test_labels, config['common']['batch_size'],
                                        val_test_transform, num_workers=config['common']['num_workers'])

    if config['common']['use_cross_validation']:
        train_with_cv(config, model,
                      train_val_file_paths=train_val_file_paths, train_val_labels= train_val_labels,
                      test_dataloader=test_dataloader,
                      train_transform=train_transform,
                      val_test_transform=val_test_transform)
    else:
        train_file_paths, val_file_paths, train_labels, val_labels = train_test_split(
            train_val_file_paths, train_val_labels, test_size=config['dataset']['val_size'], stratify=train_val_labels
        )

        train_dataloader = create_dataloader(train_file_paths, train_labels, config['common']['batch_size'],
                                             train_transform, num_workers=config['common']['num_workers'])
        val_dataloader = create_dataloader(val_file_paths, val_labels, config['common']['batch_size'],
                                           val_test_transform, num_workers=config['common']['num_workers'])

        trainer = pl.Trainer(default_root_dir=config['common']['exp_name'],
                             max_epochs=config['common'][
                                 'max_epochs'])  # You can adjust gpus parameter based on your system
        trainer.callbacks += [getattr(importlib.import_module(callback_config.target.rsplit('.', 1)[0]),
                                      callback_config.target.rsplit('.', 1)[1])(**callback_config.params) for
                              callback_config in config.callbacks]

        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, test_dataloader)

        print('The model has been successfully trained')
        exp_name = config['common'].get('exp_name', 'exp0')
        print(f'You can view training logs using the command: tensorboard --logdir {exp_name}')