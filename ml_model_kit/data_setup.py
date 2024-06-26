import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.io import read_image
import torchvision.transforms.functional as F


NUM_WORKERS = os.cpu_count()


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.classes = self.annotations["bin"].unique().tolist()

        unique_bins = sorted(self.annotations["bin"].unique())
        self.bin_to_class = {bin: i for i, bin in enumerate(unique_bins)}
        self.class_to_bin = {i: bin for bin, i in self.bin_to_class.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_name)

        image = F.convert_image_dtype(image, torch.float32)

        latitude = self.annotations.iloc[idx, 1]
        longitude = self.annotations.iloc[idx, 2]
        bin_label = self.annotations.iloc[idx, 3]
        class_number = self.bin_to_class[bin_label]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "latitude": torch.tensor(latitude, dtype=torch.float),
            "longitude": torch.tensor(longitude, dtype=torch.float),
            "bin": torch.tensor(class_number, dtype=torch.long),
        }


def get_dataset_statistics(dataset, batch_size=32, num_workers=NUM_WORKERS):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for batch in dataloader:
        images = batch["image"]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std


def create_dataloaders(
    csv_file: str,
    root_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    train_split: float,
    validation_split: float,
    num_workers: int = NUM_WORKERS,
):
    dataset_for_stats = ImageDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=None,
    )

    mean, std = get_dataset_statistics(
        dataset_for_stats,
        batch_size,
        num_workers,
    )

    transform = transforms.Compose(
        [transform, transforms.Normalize(mean=mean, std=std)]
    )

    dataset = ImageDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform,
    )

    class_names = dataset.classes

    dataset_size = len(dataset)
    train_end = int(np.floor(train_split * dataset_size))
    val_end = int(np.floor((train_split + validation_split) * dataset_size))
    indices = list(range(dataset_size))

    np.random.shuffle(indices)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    validation_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    return (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        class_names,
    )
