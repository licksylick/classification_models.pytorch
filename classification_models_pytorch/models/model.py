import os
import sys
sys.path.append('models')
import torch
import timm
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import f1_score, precision, recall
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from optimizer import Lookahead


class ClassificationModel(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_name = config['model']['params']['arch']
        try:
            self.backbone = timm.create_model(model_name,
                                              pretrained=config['model']['params']['pretrained'],
                                              num_classes=config['model']['params']['num_classes'])
        except:
            raise ValueError(f'Undefined value of model name: {model_name}')

        self.num_classes = config['model']['params']['num_classes']

    def forward(self, x):
        return self.backbone(x)

    def configure_optimizers(self):
        optimizer_name = self.config['optimizers'][0]['target']
        optimizer_params = self.config['optimizers'][0]['params']

        optimizer_class = getattr(torch.optim, optimizer_name)

        if self.config['optimizers'][0].get('use_lookahead', True):
            base_optim = optimizer_class(self.parameters(), **optimizer_params)
            optimizer = Lookahead(base_optim)
        else:
            optimizer = optimizer_class(self.parameters(), **optimizer_params)

        scheduler_name = self.config['scheduler'][0]['target']
        scheduler_params = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_params(optimizer)

        monitor = self.config['scheduler'][0].get('monitor', '')
        # return [optimizer], [scheduler]
        return  {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}

    def compute_metrics(self, pred, target):
        metrics = dict()
        metrics['f1_score'] = f1_score(pred, target, num_classes=self.num_classes, task='multiclass')
        metrics['precision'] = precision(pred, target, num_classes=self.num_classes, task='multiclass')
        metrics['recall'] = recall(pred, target, num_classes=self.num_classes, task='multiclass')
        return metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('train_f1', metrics['f1_score'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_prec', metrics['precision'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_rec', metrics['recall'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('val_f1', metrics['f1_score'], on_step=False, on_epoch=True)
        self.log('val_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('val_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('test_f1', metrics['f1_score'], on_step=False, on_epoch=True)
        self.log('test_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('test_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        print(f'test_metrics: {metrics}')

        # Convert predictions to class indices
        _, preds = torch.max(output, 1)

        return {"loss": loss, "preds": preds, "labels": y}

    def test_epoch_end(self, outputs):
        all_preds = torch.cat([out["preds"] for out in outputs])
        all_labels = torch.cat([out["labels"] for out in outputs])

        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()

        conf_matrix = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(self.num_classes, self.num_classes))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(self.num_classes), yticklabels=range(self.num_classes))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        # Save the confusion matrix as an image
        plt.savefig(os.path.join(self.config['common'].get('exp_name', 'exp0') ,"confusion_matrix.png"))
        plt.close()