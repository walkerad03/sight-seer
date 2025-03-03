import os

import torch
import torchvision
from torch import nn, optim
from torchvision import transforms


from ml_model_kit import engine, data_setup, model_builder, utils


def configure_environment():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


if __name__ == "__main__":
    configure_environment()

    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    CSV_FILE = "dataset/annotations.csv"
    ROOT_DIR = "dataset"

    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8
    VALIDATION_SPLIT = 0.2

    DATA_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    print("Setting up dataloaders")

    train_dataloader, val_dataloader, test_dataloader, class_names = (
        data_setup.create_dataloaders(
            CSV_FILE,
            ROOT_DIR,
            DATA_TRANSFORM,
            BATCH_SIZE,
            TRAIN_SPLIT,
            VALIDATION_SPLIT,
            num_workers=4,
        )
    )

    print("Creating model")

    model = model_builder.ResNet18(
        3, resblock=model_builder.ResBlock, outputs=len(class_names)
    ).to(device)

    NUM_EPOCHS = 30

    print(f"Using model: {model.name}")

    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")

    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
    )

    utils.save_model(
        model=model,
        target_dir="checkpoints",
        model_name="sightseer_deargodhelpme.pth",
    )
