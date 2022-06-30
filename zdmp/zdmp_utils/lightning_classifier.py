import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1Score


class Classifier(pl.LightningModule):
    def __init__(self, model, fine_tuning:bool= True, lr: float = 2e-5, wd:float = 0.00025, **kwargs): 
        super().__init__()
        self.save_hyperparameters('lr', 'wd', *list(kwargs))
        self.model = model
        self.fine_tuning = fine_tuning
        
        metrics = MetricCollection([
                Accuracy(num_classes=2, average='macro'), 
                Precision(num_classes=2, average='macro'), 
                Recall(num_classes=2, average='macro'), 
                F1Score(num_classes=2, average='macro')
            ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss)
        output = self.train_metrics(y_hat, y)
        self.log_dict(output)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.log("val_loss", loss)
        output = self.valid_metrics(y_hat, y)
        self.log_dict(output)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("test_loss", loss)
        output = self.test_metrics(y_hat, y)
        self.log_dict(output)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = self.hparams.wd)
        if self.fine_tuning:
            return optimizer
        
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [lr_scheduler]