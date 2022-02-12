import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy, MetricCollection, Precision, Recall


class Classifier(pl.LightningModule):
    def __init__(self, model, lr: float = 2e-5, **kwargs): 
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        
        metrics = MetricCollection([Accuracy(), Precision(), Recall()])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        
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

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=self.lr)
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay = 0.00025)