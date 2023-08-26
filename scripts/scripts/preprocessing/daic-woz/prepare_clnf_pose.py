import os
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    src_root = "./data/"
    dest_root = "../no-chunked/"
    dest_name = "head_pose.npz"
    featureID = "_CLNF_pose.txt"


    sessions = sorted(os.listdir(src_root))
    for sessionID in tqdm(sessions):
        dest_dir = os.path.join(dest_root, sessionID)
        os.makedirs(dest_dir, exist_ok=True)

        data_path = os.path.join(src_root, sessionID, sessionID+featureID)
        df = pd.read_csv(data_path, index_col=0)

        t = df.loc[:, " Tx":" Tz"].astype("float32").to_numpy()
        r = df.loc[:, " Rx":" Rz"].astype("float32").to_numpy()
        poses = np.array([t, r])
        seq = np.moveaxis(poses, 0, 1)

        dest_path = os.path.join(dest_dir, dest_name)
        np.savez_compressed(dest_path, data=seq)
