import pandas as pd
import os
import numpy as np

def get_df():
    # Enter path to the data file
    head, tail = os.path.split(os.getcwd())
    image_folder_path = os.path.join(head, "data")

    # Loads labels
    df = pd.read_csv(os.path.join(image_folder_path, "seedling_labels.csv"))

    # Creates path to top & side view
    df["color_cam_path"] = image_folder_path + "/" + df["color_cam_path"]
    df["side_cam_path"] = image_folder_path + "/" + df["side_cam_path"]

    # Gives average expert label as a starting point
    df["average_expert"] = (df["Expert 1"] + df["Expert 2"]  + df["Expert 3"] + df["Expert 4"]) / 4

    #Transform this into the binray of the task (good or poor)
    df["expert_binary"] = np.where(df["average_expert"] <= 2, 1, 0)
    return df