import os
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    src_root = "./data/"
    feature_dir = "features"
    featureID = "OpenFace2.1.0_Pose_gaze_AUs"
    dest_root = "../no-chunked/"
    dest_featureID = "video_pose_gaze_aus.npz"

    sessionIDs = sorted( os.listdir(src_root) )
    for sessionID in tqdm(sessionIDs):
        feature_path = os.path.join(src_root, sessionID, feature_dir, sessionID+"_"+featureID+".csv")
        df = pd.read_csv(feature_path)

        seq = df.iloc[:, 4:].to_numpy()

        dest_dir = os.path.join(dest_root, sessionID)
        os.makedirs(dest_dir, exist_ok = True)
        dest_path = os.path.join(dest_dir, dest_featureID)
        np.savez_compressed(dest_path, data=seq)
