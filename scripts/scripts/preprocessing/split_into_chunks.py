import os
import sys
import joblib
import numpy as np
from tqdm import tqdm

def process_video(videoID):
    dest_modality_dir = os.path.join(dest_dir, videoID.replace(".npz", ""), modality)
    os.makedirs(dest_modality_dir, exist_ok=True)

    seq_path = os.path.join(modality_dir, videoID)
    seq = np.load(seq_path)["data"]

    for start in range(0, len(seq), frame_step):
        end = min(start+frame_step, len(seq))
        chunk = seq[start:end]

        dest_path = os.path.join(dest_modality_dir, videoID.replace(".npz", "_" + str(start).zfill(6) + "_" + str(end).zfill(6) + ".npz"))
        np.savez_compressed(dest_path, data=chunk)


if __name__ == "__main__":
    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]

    nseconds = 5
    modalities = [
        ("gaze_patterns", 25),
    ]
        # ("face_emonet_embeddings", 25)
        # ("audio_pase_embeddings", 100),
        # ("body_pose_landmarks", 25),
        # ("hand_landmarks", 25),
        # ("face_landmarks", 25),
    # ]

    for modality, fps in tqdm(modalities):
        frame_step = fps * nseconds
        modality_dir = os.path.join(source_dir, modality)

        videoIDs = tqdm(sorted(os.listdir(modality_dir)), leave=False)

        joblib.Parallel(n_jobs=8)(
            joblib.delayed(process_video)(videoID) for videoID in videoIDs
        )

