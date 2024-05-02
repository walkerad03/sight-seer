import pandas as pd
import os
import shutil


def bin_data(df: pd.DataFrame, max_bin_size: int):
    if len(df) < max_bin_size:
        return df

    if "bin" not in df:
        df["bin"] = ""

    center_lat = (df["Latitude"].min() + df["Latitude"].max()) / 2
    center_lon = (df["Longitude"].min() + df["Longitude"].max()) / 2

    df.loc[
        (df["Latitude"] > center_lat) & (df["Longitude"] > center_lon), "bin"
    ] += "1"
    df.loc[
        (df["Latitude"] <= center_lat) & (df["Longitude"] > center_lon), "bin"
    ] += "2"
    df.loc[
        (df["Latitude"] > center_lat) & (df["Longitude"] <= center_lon), "bin"
    ] += "3"
    df.loc[
        (df["Latitude"] <= center_lat) & (df["Longitude"] <= center_lon), "bin"
    ] += "4"

    bin1 = bin_data(df[df["bin"].str.endswith("1")], max_bin_size=max_bin_size)
    bin2 = bin_data(df[df["bin"].str.endswith("2")], max_bin_size=max_bin_size)
    bin3 = bin_data(df[df["bin"].str.endswith("3")], max_bin_size=max_bin_size)
    bin4 = bin_data(df[df["bin"].str.endswith("4")], max_bin_size=max_bin_size)

    return pd.concat([bin1, bin2, bin3, bin4])


CSV_PATH = "raw_images/coords.csv"
IMAGES_PATH = "raw_images/"
FINAL_PATH = "dataset/"
MAX_BIN_SIZE = 50
MIN_BIN_SIZE = 48

if os.path.exists(FINAL_PATH):
    shutil.rmtree(FINAL_PATH)
os.makedirs(FINAL_PATH)


data = pd.read_csv(
    CSV_PATH,
    header=0,
    names=["filename", "Latitude", "Longitude"],
)

binned_data = bin_data(data, max_bin_size=MAX_BIN_SIZE)

bin_counts = binned_data["bin"].value_counts()
bins_to_keep = bin_counts[bin_counts >= MIN_BIN_SIZE].index
binned_data = binned_data[binned_data["bin"].isin(bins_to_keep)]

binned_data["SortKey"] = (
    binned_data["filename"].str.extract(r"(\d+)").astype(int)
)
binned_data = binned_data.sort_values(by="SortKey")
binned_data.drop("SortKey", axis=1, inplace=True)

binned_data.to_csv(f"{FINAL_PATH}/annotations.csv", index=False)
for filename in binned_data["filename"].values:
    file_path = os.path.join(IMAGES_PATH, filename)
    if filename.lower().endswith(".png"):
        dest_path = os.path.join(FINAL_PATH, filename)
        shutil.copy(file_path, dest_path)
