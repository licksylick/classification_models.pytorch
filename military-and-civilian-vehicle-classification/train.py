import os
import warnings
import pandas as pd
import pytorch_lightning as pl
from data import VehicleDataModule
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import shuffle
from utils import get_model, get_dataset_counts_info, preprocess_csv
from config import BATCH_SIZE, DATA_PATH, NUM_CLASSES

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--model', type=str, required=True, help='Model name (resnet18|resnet34|resnet50|efficientnet')
    parser.add_argument('--epoch_num', type=int, required=True, help='Epoch num')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to data directory')
    parser.add_argument('--tune', type=bool, default=False, help='Tune model before training (find LR and batch size )')
    parser.add_argument('--log_dir', type=str, default='default', help='Log directory name')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--checkpoints', type=str, required=True, help='Directory name where to save checkpoints')
    args = parser.parse_args()

    X_train = pd.read_csv(os.path.join(DATA_PATH, 'train_labels.csv'))
    X_test = pd.read_csv(os.path.join(DATA_PATH, 'test_labels.csv'))

    X_train = preprocess_csv(X_train)
    X_test = preprocess_csv(X_test)

    X_train = shuffle(X_train)
    X_test = shuffle(X_test)

    X_train.to_csv('train.csv', index=False)
    X_test.to_csv('test.csv', index=False)
    get_dataset_counts_info(X_train, X_test)

    model = get_model(args.model, NUM_CLASSES)

    dm = VehicleDataModule(data_path='train.csv', batch_size=BATCH_SIZE)

    logger = TensorBoardLogger(save_dir=os.getcwd(), name='logs')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(os.getcwd(), args.checkpoints),
        filename=f'{args.model}' + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    early_stoping_callback = EarlyStopping('val_loss', patience=5)
    trainer = pl.Trainer(accelerator='gpu', gpus=1, devices=-1, max_epochs=args.epoch_num)
    trainer.logger = logger
    trainer.callbacks = [checkpoint_callback, early_stoping_callback]
    trainer.fit(model=model, datamodule=dm)

    print('Best model with loss {:.4f} located in {}'.format(checkpoint_callback.best_model_score,
                                                             checkpoint_callback.best_model_path))

    trainer.test(model=model, datamodule=dm)

