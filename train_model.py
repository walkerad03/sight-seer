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

    CSV_FILE = "dataset/annotations.csv"
    ROOT_DIR = "dataset"

    BATCH_SIZE = 4
    VALIDATION_SPLIT = 0.2

    DATA_TRANSFORM = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )

    train_dataloader, val_dataloader, class_names = (
        data_setup.create_dataloaders(
            CSV_FILE, ROOT_DIR, DATA_TRANSFORM, BATCH_SIZE, VALIDATION_SPLIT
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
        test_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
    )

    utils.save_model(
        model=model_res,
        target_dir="unfinished_models",
        model_name="sightseer_res18_10epochs_quadtree.pth",
    )
