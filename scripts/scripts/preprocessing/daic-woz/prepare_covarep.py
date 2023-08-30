import os
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    src_root = "./data/"
    dest_root = "../no-chunked/"
    dest_name = "audio_covarep.npz"
    featureID = "_COVAREP.csv"


    sessions = sorted(os.listdir(src_root))
    for sessionID in tqdm(sessions):
        dest_dir = os.path.join(dest_root, sessionID)
        os.makedirs(dest_dir, exist_ok=True)

        data_path = os.path.join(src_root, sessionID, sessionID+featureID)
        df = pd.read_csv(data_path, header=None)

        seq = df.astype("float32").to_numpy()
        seq[np.isneginf(seq)] = 0.
        dest_path = os.path.join(dest_dir, dest_name)
        np.savez_compressed(dest_path, data=seq)
