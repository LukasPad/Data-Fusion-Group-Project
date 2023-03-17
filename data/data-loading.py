import pandas as pd
import os

# Enter path to the data_fusion_guest_lecture file
image_folder_path = "data_fusion_guest_lecture"

# Loads labels
df = pd.read_csv(os.path.join(image_folder_path, "seedling_labels.csv"))

# Creates path to top & side view
df["color_cam_path"] = image_folder_path + "/" + df["color_cam_path"]
df["side_cam_path"] = image_folder_path + "/" + df["side_cam_path"]

# Gives average expert label as a starting point
df["average_expert"] = (df["Expert 1"] + df["Expert 2"]  + df["Expert 3"] + df["Expert 4"]) / 4
df
