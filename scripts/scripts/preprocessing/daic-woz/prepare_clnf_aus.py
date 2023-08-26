import os
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    src_root = "./data/"
    dest_root = "../no-chunked/"
    dest_name = "facial_aus.npz"
    featureID = "_CLNF_AUs.txt"


    sessions = sorted(os.listdir(src_root))
    for sessionID in tqdm(sessions):
        facial_aus = []
        dest_dir = os.path.join(dest_root, sessionID)
        os.makedirs(dest_dir, exist_ok=True)

        data_path = os.path.join(src_root, sessionID, sessionID+featureID)
        df = pd.read_csv(data_path, index_col=0)

        for column_id in df.columns[3:]:
            au = df[column_id].astype("float32").to_numpy()
            facial_aus.append(au)
        seq = np.moveaxis(np.array(facial_aus), 0, -1)

        dest_path = os.path.join(dest_dir, dest_name)
        np.savez_compressed(dest_path, data=seq)
