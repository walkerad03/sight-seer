import os
import pandas as pd

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from ml_model_kit.data_setup import (
    create_dataloaders,
    ImageDataset,
)

CSV_FILE = "mock_dataset/annotations.csv"
ROOT_DIR = "mock_dataset"

transform = transforms.Compose([transforms.Resize((128, 128))])


def setup_module(module):
    data = {
        "image_path": ["0.png", "1.png"],
        "latitude": [34.0522, 36.7783],
        "longitude": [-118.2437, -119.4179],
        "bin": [0, 1],
    }
    df = pd.DataFrame(data)

    os.makedirs(ROOT_DIR, exist_ok=True)

    df.to_csv(CSV_FILE, index=False)

    for img_name in data["image_path"]:
        img = torch.randint(0, 256, (3, 600, 600), dtype=torch.uint8)
        img = img.permute(1, 2, 0).numpy()
        img = Image.fromarray(img)
        img.save(os.path.join(ROOT_DIR, img_name), format="png")


def teardown_module(module):
    os.remove(CSV_FILE)
    for img_name in os.listdir(ROOT_DIR):
        os.remove(os.path.join(ROOT_DIR, img_name))
    os.rmdir(ROOT_DIR)


def test_image_dataset_init():
    dataset = ImageDataset(CSV_FILE, ROOT_DIR, transform=transform)
    assert len(dataset) == 2


def test_image_dataset_getitem():
    dataset = ImageDataset(
        csv_file=CSV_FILE, root_dir=ROOT_DIR, transform=transform
    )
    sample = dataset[0]
    assert "image" in sample, sample
    assert "targets" in sample
    assert sample["targets"][0] == torch.tensor(34.0522)
    assert sample["targets"][1] == torch.tensor(-118.2437)


def test_create_dataloaders():
    train_dataloader, val_dataloader, test_dataloader, class_names = (
        create_dataloaders(
            csv_file=CSV_FILE,
            root_dir=ROOT_DIR,
            transform=transform,
            batch_size=2,
            train_split=0.5,
            validation_split=0.25,
        )
    )

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(val_dataloader, DataLoader)
    assert isinstance(test_dataloader, DataLoader)
    assert isinstance(class_names, list)
    assert len(class_names) == 2  # Matches the bins in our mock data
    assert len(train_dataloader.dataset) > 0
    assert len(val_dataloader.dataset) > 0
    assert len(test_dataloader.dataset) > 0
