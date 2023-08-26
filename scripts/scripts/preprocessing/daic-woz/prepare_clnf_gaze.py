import os
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    src_root = "./data/"
    dest_root = "../no-chunked/"
    dest_name = "gaze.npz"
    featureID = "_CLNF_gaze.txt"


    sessions = sorted(os.listdir(src_root))
    for sessionID in tqdm(sessions):
        dest_dir = os.path.join(dest_root, sessionID)
        os.makedirs(dest_dir, exist_ok=True)

        data_path = os.path.join(src_root, sessionID, sessionID+featureID)
        df = pd.read_csv(data_path, index_col=0)

        v0 = df.loc[:, " x_0":" z_0"].astype("float32").to_numpy()
        v1 = df.loc[:, " x_1":" z_1"].astype("float32").to_numpy()
        vh0 = df.loc[:, " x_h0":" z_h0"].astype("float32").to_numpy()
        vh1 = df.loc[:, " x_h1":" z_h1"].astype("float32").to_numpy()

        gazes = np.array([v0, v1, vh0, vh1])
        seq = np.moveaxis(gazes, 0, 1)

        dest_path = os.path.join(dest_dir, dest_name)
        np.savez_compressed(dest_path, data=seq)
