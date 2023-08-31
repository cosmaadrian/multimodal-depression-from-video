import os
import scipy.io
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    src_root = "./data/"
    feature_dir = "features"
    featureID = "CNN_VGG"
    dest_root = "../no-chunked/"
    dest_featureID = "video_cnn_vgg.npz"

    sessionIDs = sorted( os.listdir(src_root) )
    for sessionID in tqdm(sessionIDs):
        if sessionID != "657":
            feature_path = os.path.join(src_root, sessionID, feature_dir, sessionID+"_"+featureID+".mat")
            data = scipy.io.loadmat(feature_path)

            seq = data["feature"]

            dest_dir = os.path.join(dest_root, sessionID)
            os.makedirs(dest_dir, exist_ok = True)
            dest_path = os.path.join(dest_dir, dest_featureID)
            np.savez_compressed(dest_path, data=seq)
