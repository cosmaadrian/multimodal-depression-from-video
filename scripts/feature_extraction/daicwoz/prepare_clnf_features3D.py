import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-root", type=str, default="./data/DAIC-WOZ/data/")
    parser.add_argument("--modality-id", type=str, default="facial_3d_landmarks")
    parser.add_argument("--dest-root", type=str, default="./data/DAIC-WOZ/no-chunked/")
    args = parser.parse_args()

    featureID = "_CLNF_features3D.txt"

    dest_dir = os.path.join(args.dest_root, args.modality_id)
    os.makedirs(dest_dir, exist_ok=True)

    sessions = sorted(os.listdir(args.src_root))
    for sessionID in tqdm(sessions):
        facial_3d_landmarks = []

        data_path = os.path.join(args.src_root, sessionID, sessionID+featureID)
        df = pd.read_csv(data_path, index_col=0)

        for i in range(0, 68):
            x = df[f" X{i}"].astype("float32").to_numpy()
            y = df[f" Y{i}"].astype("float32").to_numpy()
            z = df[f" Z{i}"].astype("float32").to_numpy()
            landmark = np.vstack((x,y,z))
            facial_3d_landmarks.append(landmark)

        dest_path = os.path.join(dest_dir, sessionID+".npz")
        seq = np.moveaxis(np.array(facial_3d_landmarks), -1, 0)
        np.savez_compressed(dest_path, data=seq)
