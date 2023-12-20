import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-root", type=str, default="./data/E-DAIC/data/")
    parser.add_argument("--modality_id", type=str, default="video_pose_gaze_aus")
    parser.add_argument("--dest-root", type=str, default="./data/E-DAIC/no-chunked/")
    args = parser.parse_args()

    feature_dir = "features"
    featureID = "OpenFace2.1.0_Pose_gaze_AUs"

    dest_dir = os.path.join(args.dest_root, args.modality_id)
    os.makedirs(dest_dir, exist_ok = True)

    sessionIDs = sorted( os.listdir(args.src_root) )
    for sessionID in tqdm(sessionIDs):
        feature_path = os.path.join(args.src_root, sessionID, feature_dir, sessionID+"_"+featureID+".csv")
        df = pd.read_csv(feature_path)

        seq = df.iloc[:, 4:].to_numpy()

        dest_path = os.path.join(dest_dir, sessionID+".npz")
        np.savez_compressed(dest_path, data=seq)
