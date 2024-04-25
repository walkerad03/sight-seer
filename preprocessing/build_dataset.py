import pandas as pd
import os
import shutil
from tqdm import tqdm

COORDS_PATH = "raw_images/coords.csv"
IMAGES_PATH = "raw_images/"
FINAL_PATH = "dataset/"
BIN_SIZE = 5  # Degrees
TRAIN_TEST_RATIO = 0.85


def _calculate_bin(latitude, longitude, granularity):
    """
    Compute a class for a given latitude and longitude.

    Bins are grid squares spread across any point on the globe. `granularity`
    defines the width of the squares in degrees.
    """
    lat_bin = int((latitude + 90) / granularity)
    lon_bin = int((longitude + 180) / granularity)
    num_lon_bins = int(360 / granularity)
    return lat_bin * num_lon_bins + lon_bin


# Cleanup and preparation
if os.path.exists(FINAL_PATH):
    shutil.rmtree(FINAL_PATH)
os.makedirs(FINAL_PATH)

train_path = os.path.join(FINAL_PATH, "train")
test_path = os.path.join(FINAL_PATH, "test")
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Load coordinates and calculate bins
coords_df = pd.read_csv(COORDS_PATH, header=0, names=["Latitude", "Longitude"])
coords_df["Bin"] = coords_df.apply(
    lambda row: _calculate_bin(row["Latitude"], row["Longitude"], BIN_SIZE),
    axis=1,
)

# Track bins that have images in both train and test
train_bins = set()
test_bins = set()

train_cutoff = int(len(coords_df) * TRAIN_TEST_RATIO)

# Image distribution
for index, row in tqdm(coords_df.iterrows(), total=coords_df.shape[0]):
    placement_folder = train_path if index <= train_cutoff else test_path
    placement_set = train_bins if index <= train_cutoff else test_bins
    image_name = f"{index}.png"
    bin_folder = str(int(row["Bin"]))
    bin_folder_path = os.path.join(placement_folder, bin_folder)

    os.makedirs(bin_folder_path, exist_ok=True)
    shutil.copy(
        os.path.join(IMAGES_PATH, image_name),
        os.path.join(bin_folder_path, image_name),
    )
    placement_set.add(bin_folder)

# Remove bins not in both train and test
for bin_folder in train_bins | test_bins:
    if bin_folder not in train_bins or bin_folder not in test_bins:
        shutil.rmtree(os.path.join(train_path, bin_folder), ignore_errors=True)
        shutil.rmtree(os.path.join(test_path, bin_folder), ignore_errors=True)

# Save the modified coordinates DataFrame
coords_df.to_csv(os.path.join(FINAL_PATH, "coords_modified.csv"))
