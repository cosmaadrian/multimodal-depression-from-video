import os
import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":
    split_path = sys.argv[1]

    split_df = pd.read_csv(split_path, index_col=0)
    no_modality_idxs_files = [
        ("no_voice_idxs.npz", "voice_presence"),
        ("no_face_idxs.npz", "face_presence"),
        ("no_body_idxs.npz", "body_presence"),
        ("no_hand_idxs.npz", "hand_presence"),
    ]

    for no_modality_idxs_file, column_id in no_modality_idxs_files:
        modality_seconds = []
        for videoID in split_df["video_id"].tolist():
            no_modality_idxs_path = os.path.join("./data/", videoID, no_modality_idxs_file)
            no_modality_idxs = np.load(no_modality_idxs_path)["data"]

            video_duration = split_df[split_df["video_id"] == videoID]["duration"].values[0]

            if "voice" in no_modality_idxs_file:
                frame_rate = split_df[split_df["video_id"] == videoID]["audio_frame_rate"].values[0]
            else:
                frame_rate = split_df[split_df["video_id"] == videoID]["video_frame_rate"].values[0]

            modality_second = video_duration - (no_modality_idxs.shape[0] / frame_rate)
            modality_seconds.append(modality_second)
            # print(videoID, video_duration, no_modality_idxs_file, no_modality_idxs.shape, frame_rate, modality_seconds)
        split_df[column_id] = modality_seconds
    print(split_df)
    split_df.to_csv(split_path.replace(".csv", "_2.csv"))
