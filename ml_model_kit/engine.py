from typing import Tuple, Dict, List
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import os
import pandas as pd
import math

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

    for batch, (X, y) in enumerate(dataloader):
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
        for batch, (X, y) in enumerate(dataloader):
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


def _create_lat_lon_tensors(num_bins, granularity):
    latitudes = []
    longitudes = []
    for bin_index in range(num_bins):
        lat, lon = _calculate_lat_lon_from_bin(bin_index, granularity)
        latitudes.append(lat)
        longitudes.append(lon)
    return torch.tensor(latitudes), torch.tensor(longitudes)


def _calculate_lat_lon_from_bin(bin_index, granularity):
    num_lon_bins = int(360 / granularity)
    lat_bin = bin_index // num_lon_bins
    lon_bin = bin_index % num_lon_bins

    latitude = lat_bin * granularity - 90 + granularity / 2
    longitude = lon_bin * granularity - 180 + granularity / 2

    return (latitude, longitude)


def _calculate_distance(lat1, lon1, lat2, lon2):
    # Calculate distance using the haversine formula.
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    r = 6371

    return c * r


def _create_lat_lon_tensors_for_classes(dataloader, granularity):
    """
    Create tensors for latitude and longitude only for classes present in the
    dataloader.
    """
    class_indices = dataloader.dataset.class_to_idx.values()
    latitudes = []
    longitudes = []

    for idx in class_indices:
        lat, lon = _calculate_lat_lon_from_bin(int(idx), granularity)
        latitudes.append(lat)
        longitudes.append(lon)

    return torch.tensor(latitudes, dtype=torch.float32), torch.tensor(
        longitudes, dtype=torch.float32
    )


def eval_tiered_accuracy(
    model: nn.Module,
    eval_dataloader: DataLoader,
    device: torch.device,
    granularity: int = 20,
) -> Tuple[float, float, float, float, float]:
    """
    NOTE: This is currently no different to the test_step function. It will
    eventually test the model at various accuracy ranges using the following
    procedure.

    1. Perform a weighted average between the model's logits and the
    geographical centerpoints of corresponding bins. This will be the final
    output as a latitude and longitude.

    2. Check the distance from this output to the true location.

    3. Do this for every image in the test set, and compute a success rate at
    the following accuracy ranges:
    a. Street (1km)
    b. City (25km)
    c. Region (200 km)
    d. Country (750 km)
    e. Continent (2500 km)
    """
    model.eval()

    coords_df = pd.read_csv("processed/coords_modified.csv")

    accuracy_ranges = [1, 25, 200, 750, 2500]
    accuracies = [0, 0, 0, 0, 0]
    total_samples = 0

    latitudes, longitudes = _create_lat_lon_tensors_for_classes(
        eval_dataloader, granularity
    )
    latitudes, longitudes = latitudes.to(device), longitudes.to(device)

    with torch.inference_mode():
        for batch, (X, _, paths) in enumerate(eval_dataloader):
            X = X.to(device)

            test_pred_logits = model(X)
            softmax_probs = nn.functional.softmax(test_pred_logits, dim=1)

            weighted_lat = (softmax_probs * latitudes).sum(dim=1)
            weighted_lon = (softmax_probs * longitudes).sum(dim=1)

            for i, path in enumerate(paths):
                index = int(os.path.splitext(os.path.basename(path))[0])
                predicted_lat = weighted_lat[i]
                predicted_lon = weighted_lon[i]
                true_lat = coords_df.loc[index, "Latitude"]
                true_lon = coords_df.loc[index, "Longitude"]

                distance = _calculate_distance(
                    predicted_lat, predicted_lon, true_lat, true_lon
                )

                for j, range_km in enumerate(accuracy_ranges):
                    if distance <= range_km:
                        accuracies[j] += 1

            total_samples += len(weighted_lat)

    accuracies = [acc / total_samples for acc in accuracies]

    return tuple(accuracies)


def make_single_prediction(model, dataloader, image_tensor, device):
    """
    Take an image input as a tensor, and run it through the model in inference mode.

    Return a latitude and longitude
    """
    model.eval()

    latitudes, longitudes = _create_lat_lon_tensors_for_classes(dataloader, 5)
    latitudes, longitudes = latitudes.to(device), longitudes.to(device)

    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        test_pred_logits = model(image_tensor)
        softmax_probs = nn.functional.softmax(test_pred_logits, dim=1)

        weighted_lat = (softmax_probs * latitudes).sum(dim=1)
        weighted_lon = (softmax_probs * longitudes).sum(dim=1)

    return weighted_lat.item(), weighted_lon.item()
