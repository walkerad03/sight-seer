import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.io import read_image


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

        try:
            image = read_image(img_name)
        except Exception as e:
            print(f"Error loading image {img_name}:\n{e}\n")
            return None

        try:
            targets = torch.tensor(
                [
                    float(self.annotations.iloc[idx, 1]),
                    float(self.annotations.iloc[idx, 2]),
                ],
                dtype=torch.float32,
            )
        except Exception as e:
            print(f"Error processing targets for idx {idx}:\n{e}\n")
            return {
                "error": f"Target processing failed for idx {idx}",
                "idx": idx,
            }

        if self.transform:
            image = self.transform(image.float())

        return {"image": image, "targets": targets}


def create_dataloaders(
    csv_file: str,
    root_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    train_split: float,
    validation_split: float,
    num_workers: int = NUM_WORKERS,
):
    mean = torch.Tensor([137.1102, 144.0311, 137.1939])
    std = torch.Tensor([46.3285, 45.0499, 60.7456])

    transform = transforms.Compose(
        [
            transform,
            transforms.Normalize(mean=mean, std=std),
        ]
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
        sampler=train_sampler,
    )

    validation_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_sampler,
    )

    test_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=test_sampler,
    )

    return (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        class_names,
    )
