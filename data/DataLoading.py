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
    exp_binary = []
    for x, y in df[["Expert 1", "Expert 2", "Expert 3", "Expert 4"]].iterrows():
        healthy = 0
        unhealthy = 0
        for z in y:
            if z > 2:
                unhealthy += 1
            else:
                healthy += 1
        if healthy > 2:
            exp_binary.append(1)
        elif unhealthy > 2:
            exp_binary.append(0)
        else:
            exp_binary.append(np.NaN)
    df["binary_expert"] = exp_binary
    df = df.dropna()
    return df


if __name__ == "__main__":
    df = get_df()
