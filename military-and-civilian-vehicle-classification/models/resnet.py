import torch
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn.functional as F
from torchmetrics.functional import accuracy, precision, recall
from torch import nn
from optimizer import Lookahead


class Resnet(pl.LightningModule):

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError(f'Undefined value of model name: {model_name}')

        self.num_classes = num_classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    def configure_optimizers(self):
        base_optim = torch.optim.RAdam(self.parameters())
        optimizer = Lookahead(base_optim)
        return [optimizer]

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
        self.log('train_acc', metrics['accuracy'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_prec', metrics['precision'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_rec', metrics['recall'], prog_bar=True, on_step=False, on_epoch=True)
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
        print(f'test_metrics: {metrics}')
        return loss
