from torchvision import transforms
from torch.utils.data import DataLoader

from ml_model_kit.data_setup import ImageDataset

from tqdm import tqdm

CSV_FILE_PATH = "dataset/annotations.csv"
IMAGE_ROOT_DIR = "dataset"


def test_dataloader():
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    dataset = ImageDataset(CSV_FILE_PATH, IMAGE_ROOT_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4)

    try:
        for batch in tqdm(dataloader):
            if batch is None:
                print("Skipped a batch due to NoneType return.")
                continue
            print("Loaded batch:")
            print("Images shape:", batch["image"].shape)
            print("Latitude shape:", batch["latitude"].shape)
            print("Longitude shape:", batch["longitude"].shape)
            # break  # Load only the first batch for testing
    except Exception as e:
        print(f"Error during DataLoader iteration:\n{e}")


if __name__ == "__main__":
    test_dataloader()
