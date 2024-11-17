import os
import torch
from torchvision import transforms
from ml_model_kit import model_builder
from PIL import Image


def configure_environment():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


if __name__ == "__main__":
    configure_environment()

    MODEL_PATH = "unfinished_models/sightseer_res18_10epochs_regression.pth"
    IMAGE_PATH = "test05.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_builder.ResNet18(
        3, resblock=model_builder.ResBlock, outputs=151
    )

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[137.1102, 144.0311, 137.1939],
                std=[46.3285, 45.0499, 60.7456],
            ),
        ]
    )

    image = Image.open(IMAGE_PATH).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.inference_mode():
        pred = model(image)

    print(pred.cpu().numpy())
