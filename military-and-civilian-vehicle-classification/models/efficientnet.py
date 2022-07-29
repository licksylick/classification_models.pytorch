import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy, precision, recall
from torch import nn, optim
from efficientnet_pytorch import EfficientNet
from optimizer import Lookahead


class EfficientNetModel(pl.LightningModule):

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    def configure_optimizers(self):
        base_optim = torch.optim.Adam(self.parameters())
        optimizer = Lookahead(base_optim)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [6, 9])

        return [optimizer], [scheduler]

    def compute_metrics(self, pred, target):
        metrics = dict()
        metrics['accuracy'] = accuracy(pred, target, num_classes=self.num_classes)
        metrics['precision'] = precision(pred, target, num_classes=self.num_classes)
        metrics['recall'] = recall(pred, target, num_classes=self.num_classes)

        return metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('train_acc', metrics['accuracy'], on_step=False, on_epoch=True)
        self.log('train_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('train_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('val_acc', metrics['accuracy'], on_step=False, on_epoch=True)
        self.log('val_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('val_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('test_acc', metrics['accuracy'], on_step=False, on_epoch=True)
        self.log('test_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('test_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
