import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-root", type=str, default="./data/DAIC-WOZ/data/")
    parser.add_argument("--modality-id", type=str, default="gaze")
    parser.add_argument("--dest-root", type=str, default="./data/DAIC-WOZ/no-chunked/")
    args = parser.parse_args()

    featureID = "_CLNF_gaze.txt"

    dest_dir = os.path.join(args.dest_root, args.modality_id)
    os.makedirs(dest_dir, exist_ok=True)

    sessions = sorted(os.listdir(args.src_root))
    for sessionID in tqdm(sessions):

        data_path = os.path.join(args.src_root, sessionID, sessionID+featureID)
        df = pd.read_csv(data_path, index_col=0)

        v0 = df.loc[:, " x_0":" z_0"].astype("float32").to_numpy()
        v1 = df.loc[:, " x_1":" z_1"].astype("float32").to_numpy()
        vh0 = df.loc[:, " x_h0":" z_h0"].astype("float32").to_numpy()
        vh1 = df.loc[:, " x_h1":" z_h1"].astype("float32").to_numpy()

        gazes = np.array([v0, v1, vh0, vh1])
        seq = np.moveaxis(gazes, 0, 1)

        dest_path = os.path.join(dest_dir, sessionID+".npz")
        np.savez_compressed(dest_path, data=seq)
