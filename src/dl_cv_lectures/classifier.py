"""Image Classification Networks."""
import lightning as L
import torch
import torch.nn as nn
import torchmetrics


class Classifier(L.LightningModule):
    """Lightning Module to Track a generic classification model."""

    def __init__(self, model, num_classes: int, weight_decay: float = 0.0):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.weight_decay = weight_decay

        # Define individual metrics
        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.train_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes
        )

        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=num_classes
        )

        # Track the losses
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.train_loss.update(loss)
        self.train_accuracy.update(preds, y)
        self.train_f1.update(preds, y)

        # Log metrics for this batch
        self.log("train/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(
            "train/accuracy_step", self.train_accuracy, prog_bar=True, on_step=True, on_epoch=False
        )
        self.log("train/f1_step", self.train_f1, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def on_train_epoch_end(self):
        # Log average loss and metrics for the entire epoch
        avg_loss = self.train_loss.compute()
        avg_accuracy = self.train_accuracy.compute()
        avg_f1 = self.train_f1.compute()

        self.log("train/loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
        self.log("train/accuracy_epoch", avg_accuracy, prog_bar=True, on_epoch=True)
        self.log("train/f1_epoch", avg_f1, prog_bar=True, on_epoch=True)

        # Reset metrics for the next epoch
        self.train_loss.reset()
        self.train_accuracy.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_loss.update(loss)
        self.val_accuracy.update(preds, y)
        self.val_f1.update(preds, y)

        # Log metrics for this batch
        self.log("val/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(
            "val/accuracy_step", self.val_accuracy, prog_bar=True, on_step=True, on_epoch=False
        )
        self.log("val/f1_step", self.val_f1, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def on_validation_epoch_end(self):
        # Log average loss and metrics for the entire validation epoch
        avg_loss = self.val_loss.compute()
        avg_accuracy = self.val_accuracy.compute()
        avg_f1 = self.val_f1.compute()

        self.log("val/loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
        self.log("val/accuracy_epoch", avg_accuracy, prog_bar=True, on_epoch=True)
        self.log("val/f1_epoch", avg_f1, prog_bar=True, on_epoch=True)

        # Reset metrics for the next epoch
        self.val_loss.reset()
        self.val_accuracy.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=self.weight_decay)
