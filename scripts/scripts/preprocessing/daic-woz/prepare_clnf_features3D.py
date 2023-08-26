import os
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    src_root = "./data/"
    dest_root = "../no-chunked/"
    dest_name = "facial_3d_landmarks.npz"
    featureID = "_CLNF_features3D.txt"

    sessions = sorted(os.listdir(src_root))
    for sessionID in tqdm(sessions):
        facial_2d_landmarks = []
        dest_dir = os.path.join(dest_root, sessionID)
        os.makedirs(dest_dir, exist_ok=True)

        data_path = os.path.join(src_root, sessionID, sessionID+featureID)
        df = pd.read_csv(data_path, index_col=0)

        for i in range(0, 68):
            x = df[f" X{i}"].astype("float32").to_numpy()
            y = df[f" Y{i}"].astype("float32").to_numpy()
            z = df[f" Z{i}"].astype("float32").to_numpy()
            landmark = np.vstack((x,y,z))
            facial_2d_landmarks.append(landmark)

        dest_path = os.path.join(dest_dir, dest_name)
        seq = np.moveaxis(np.array(facial_2d_landmarks), -1, 0)
        np.savez_compressed(dest_path, data=seq)
