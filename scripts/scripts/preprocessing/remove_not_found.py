import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    source_dir = sys.argv[1]
    not_found_df_path = sys.argv[2]

    not_found = pd.read_csv("../not_found.csv")["video_id"].tolist()
    samples = sorted(os.listdir(source_dir))
    for sample in tqdm(samples):
        sampleID = sample.split(".")[0]
        if sampleID in not_found:
            sample_path = os.path.join(source_dir, sample)
            # data = np.load(sample_path)["data"]
            os.remove(sample_path)
