from typing import Tuple, Dict, List
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


import datetime


def _timedelta_to_hms(timedelta: datetime.timedelta) -> str:
    total_seconds = timedelta.seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y, _) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y)).sum().item() / len(
                test_pred_labels
            )

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    time_start = datetime.datetime.now()

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        time_end = datetime.datetime.now()
        time_elapsed = time_end - time_start

        epochs_remaining = epochs - 1 - epoch
        time_per_epoch = time_elapsed / (epoch + 1)
        time_remaining = time_per_epoch * epochs_remaining

        time_elapsed_string = _timedelta_to_hms(time_elapsed)
        time_remaining_string = _timedelta_to_hms(time_remaining)

        print(
            f"Epoch: {epoch+1:0{len(str(epochs))}}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc*100:.2f}% | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc*100:.2f}% | "
            f"[{time_elapsed_string}<{time_remaining_string}]"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
