import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-root", type=str, default="./data/DAIC-WOZ/data/")
    parser.add_argument("--modality-id", type=str, default="audio_covarep")
    parser.add_argument("--dest-root", type=str, default="./data/DAIC-WOZ/no-chunked/")
    args = parser.parse_args()

    featureID = "_COVAREP.csv"
    dest_dir = os.path.join(args.dest_root, args.modality_id)
    os.makedirs(dest_dir, exist_ok=True)

    sessions = sorted(os.listdir(args.src_root))
    for sessionID in tqdm(sessions):
        data_path = os.path.join(args.src_root, sessionID, sessionID+featureID)
        df = pd.read_csv(data_path, header=None)

        seq = df.astype("float32").to_numpy()
        seq[np.isneginf(seq)] = 0.

        dest_path = os.path.join(dest_dir, sessionID+".npz")
        np.savez_compressed(dest_path, data=seq)
