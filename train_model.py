from pathlib import Path

import torch
import torchvision
from torch import nn, optim
from torchvision import transforms

from ml_model_kit import engine, data_setup, model_builder, utils

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(0 / 0)

    BATCH_SIZE = 16

    data_path = Path("dataset/")

    train_dir = data_path / "train"
    test_dir = data_path / "test"

    data_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )

    train_dataloader, test_dataloader, _, class_names = (
        data_setup.create_dataloaders(
            train_dir, test_dir, data_transform, BATCH_SIZE
        )
    )

    model_res = model_builder.ResNet18(
        3, resblock=model_builder.ResBlock, outputs=len(class_names)
    ).to(device)

    NUM_EPOCHS = 10

    print(f"Using model: {model_res.name}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model_res.parameters(), lr=0.0001)

    results = engine.train(
        model=model_res,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
    )

    utils.save_model(
        model=model_res,
        target_dir="unfinished_models",
        model_name="model.pth",
    )
