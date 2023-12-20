import os
import argparse
import scipy.io
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-root", type=str, default="./data/E-DAIC/data/")
    parser.add_argument("--modality-id", type=str, default="video_cnn_resnet")
    parser.add_argument("--dest-root", type=str, default="./data/E-DAIC/no-chunked/")
    args = parser.parse_args()

    feature_dir = "features"
    featureID = "CNN_ResNet"

    dest_dir = os.path.join(args.dest_root, args.modality_id)
    os.makedirs(dest_dir, exist_ok = True)

    sessionIDs = sorted( os.listdir(args.src_root) )
    for sessionID in tqdm(sessionIDs):
        feature_path = os.path.join(args.src_root, sessionID, feature_dir, sessionID+"_"+featureID+".mat")
        data = scipy.io.loadmat(feature_path)

        seq = data["feature"]

        dest_path = os.path.join(dest_dir, sessionID+".npz")
        np.savez_compressed(dest_path, data=seq)
