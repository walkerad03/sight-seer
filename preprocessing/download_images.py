import requests
import random
import os
import pandas as pd
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Constants
API_KEY = os.getenv("GOOGLE_API_KEY")
CROP_SIZE = (512, 512)
IMAGE_SIZE = "640x640"  # Format: widthxheight
LATITUDE_BOUNDS = (
    24.396308,
    49.384358,
)  # Approximate latitude bounds for the contiguous US
LONGITUDE_BOUNDS = (
    -124.848974,
    -66.93457,
)  # Approximate longitude bounds for the contiguous US
NUM_IMAGES = 20000  # Number of images to generate
OUTPUT_DIR = "street_view_images"  # Directory to save images
CSV_FILENAME = "coords.csv"

START_AT_INDEX = 0


def get_random_us_coordinate():
    """Generate random latitude and longitude within specified bounds."""
    lat = random.uniform(*LATITUDE_BOUNDS)
    lon = random.uniform(*LONGITUDE_BOUNDS)
    return lat, lon


def check_street_view_availability(lat, lon) -> tuple:
    """
    Check if a Street View image is available at the given location using the
    Metadata API.
    """
    base_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "key": API_KEY}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["status"] == "OK":
            return True, data["location"]["lat"], data["location"]["lng"]
    return False, None, None


def get_street_view_image(lat, lon, heading):
    """
    Construct URL for Google Street View API to get an image based on given
    coordinates and heading.
    """
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": IMAGE_SIZE,
        "location": f"{lat},{lon}",
        "heading": heading,
        "key": API_KEY,
        "radius": 50,
        "source": "outdoor",
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.content
    return None


def crop_image(image_data):
    """Crop the image to a centered 512x512 square."""
    with Image.open(io.BytesIO(image_data)) as img:
        width, height = img.size  # Should be 640x640 as per IMAGE_SIZE
        left = (width - CROP_SIZE[0]) // 2
        top = (height - CROP_SIZE[1]) // 2
        right = left + CROP_SIZE[0]
        bottom = top + CROP_SIZE[1]
        cropped_img = img.crop((left, top, right, bottom))
        return cropped_img


def save_image(image, index):
    """Save the image data to a file in the specified output directory."""
    if image:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, f"{index}.png")
        image.save(file_path)
        return file_path
    return None


def main():

    coordinate_dataframe = pd.DataFrame(columns=["Latitude", "Longitude"])
    i = START_AT_INDEX

    while i < NUM_IMAGES + START_AT_INDEX:
        lat, lon = get_random_us_coordinate()

        sv_available, true_lat, true_lon = check_street_view_availability(
            lat, lon
        )

        if not sv_available:
            continue

        heading = random.randint(0, 360)
        image_data = get_street_view_image(lat, lon, heading)
        cropped_image = crop_image(image_data)
        file_path = save_image(cropped_image, i)
        if file_path:
            print(f"Saved Street View Image at: {file_path}")

        coordinate_dataframe.loc[len(coordinate_dataframe.index)] = [
            true_lat,
            true_lon,
        ]

        coordinate_dataframe.to_csv(
            f"{OUTPUT_DIR}/{CSV_FILENAME}", index=False
        )

        i += 1


if __name__ == "__main__":
    main()
