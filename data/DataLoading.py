import pandas as pd
import os
import numpy as np

def get_df(filepath :str = "seedling_labels.csv"):

    # Enter path to the data file
    image_folder_path, tail = os.path.split(filepath)
    if not len(image_folder_path) == 0:
        image_folder_path += "\\"
    # image_folder_path = os.path.join(head, "data")

    # Loads labels
    df = pd.read_csv(filepath)

    # Creates path to top & side view
    df["color_cam_path"] = image_folder_path + df["color_cam_path"]
    df["side_cam_path"] = image_folder_path + df["side_cam_path"]

    # Gives average expert label as a starting point
    df["average_expert"] = (df["Expert 1"] + df["Expert 2"]  + df["Expert 3"] + df["Expert 4"]) / 4

    #Transform this into the binray of the task (good or poor)
    df["expert_binary"] = np.where(df["average_expert"] <= 2, 1, 0)
    return df


if __name__ == "__main__":
    df = get_df()
