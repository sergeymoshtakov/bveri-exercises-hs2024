"""Image Classification Networks."""
from typing import Callable

import lightning as L
import torch
import torch.nn as nn
import torchmetrics

from tqdm.notebook import tqdm


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


def train_one_epoch(
    data_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    optimizer: torch.optim.Adam,
    loss_fn: Callable,
    device: str = "cpu",
    verbose: bool = True,
):

    net = net.to(device)

    with tqdm(data_loader, unit="batch", disable=not verbose) as tepoch:

        total_samples_seen = 0
        total_correct = 0

        for step, (X, y) in enumerate(tepoch):

            # Update Step
            logits = net(X.to(device))
            loss = loss_fn(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate Accuracy
            class_probabilities = torch.softmax(logits, axis=-1).detach().cpu()
            y_hat = (
                class_probabilities.argmax(dim=1, keepdim=True).squeeze().detach().cpu()
            )

            num_correct = (y_hat == y).sum().item()
            num_samples = X.shape[0]
            batch_accuracy = num_correct / num_samples

            # Epoch Statistics
            total_samples_seen += num_samples
            total_correct += num_correct
            epoch_accuracy = total_correct / total_samples_seen

            if verbose:
                tepoch.set_postfix(
                    loss=loss.item(),
                    accuracy_batch=batch_accuracy,
                    accuracy_epoch=epoch_accuracy,
                )


def eval_loop(
    data_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    loss_fn: Callable,
    device: str = "cpu",
) -> tuple[float, torch.Tensor, torch.Tensor]:

    net = net.to(device)
    net.eval()
    with tqdm(data_loader, unit="batch") as tepoch:

        total_samples_seen = 0
        total_correct = 0

        y_list = list()
        y_hat_list = list()

        for step, (X, y) in enumerate(tepoch):

            # Forward Pass
            with torch.no_grad():
                logits = net(X.to(device))
            loss = loss_fn(logits, y.to(device))

            # Predictions
            class_probabilities = torch.softmax(logits, axis=-1).detach().cpu()
            y_hat = (
                class_probabilities.argmax(dim=1, keepdim=True).squeeze().detach().cpu()
            )

            # Metrics
            num_correct = (y_hat == y).sum().item()
            num_samples = X.shape[0]
            total_samples_seen += num_samples
            total_correct += num_correct
            epoch_accuracy = total_correct / total_samples_seen

            tepoch.set_postfix(
                loss=loss.item(),
                accuracy_epoch=epoch_accuracy,
            )

            # save preds and targets
            y_list.append(y.cpu())
            y_hat_list.append(y_hat.cpu())

    return epoch_accuracy, torch.concat(y_list), torch.concat(y_hat_list)
