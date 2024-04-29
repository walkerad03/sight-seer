import os
import io
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


NUM_WORKERS = os.cpu_count()


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.classes = self.annotations["bin"].unique().tolist()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = io.imread(img_name)
        annotations = self.annotations.iloc[idx, 1:]
        annotations = np.array([annotations]).reshape(-1, 2)
        sample = {"image": image, "annotations": annotations}

        if self.transform:
            sample = self.transform(sample)

        return sample


def create_dataloaders(
    csv_file: str,
    root_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    validation_split: float,
    num_workers: int = NUM_WORKERS,
):
    dataset = ImageDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform,
    )

    class_names = dataset.classes

    dataset_size = len(dataset)
    split = int(np.floor(validation_split * dataset_size))
    indices = list(range(dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

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

    return train_dataloader, validation_dataloader, class_names
