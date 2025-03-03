from typing import Tuple, Dict, List
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ml_model_kit import utils

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
    train_loss = 0

    for batch in dataloader:
        X, y = (
            batch["image"].to(device),
            batch["targets"].to(device),
        )

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(dataloader)
    return train_loss


def val_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    val_loss = 0

    with torch.inference_mode():
        for batch in dataloader:
            X, y = (
                batch["image"].to(device),
                batch["targets"].to(device),
            )

            val_pred = model(X)

            loss = loss_fn(val_pred, y)

            val_loss += loss.item()

    val_loss /= len(dataloader)
    return val_loss


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    results = {
        "train_loss": [],
        "val_loss": [],
    }

    time_start = datetime.datetime.now()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/training_{current_time}"
    writer = SummaryWriter(log_dir)

    sample_input = next(iter(train_dataloader))["image"].to(device)
    writer.add_graph(model, sample_input)

    for epoch in range(epochs):
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_loss = val_step(
            model=model,
            dataloader=val_dataloader,
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
            f"val_loss: {val_loss:.4f} | "
            f"[{time_elapsed_string}<{time_remaining_string}]"
        )

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        writer.add_scalars(
            "loss",
            {
                "train": train_loss,
                "val": val_loss,
            },
            epoch + 1,
        )

        if epoch % 10 == 0:
            utils.save_model(
                model=model,
                target_dir="checkpoints",
                model_name=f"sightseer_deargodhelpme_step{epoch}.pth",
            )

    writer.close()
    return results
