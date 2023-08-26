import os
import sys
import numpy as np
from tqdm import tqdm

def process_video(sessionID, modality):
    dest_modality_dir = os.path.join(dest_dir, sessionID, modality)
    os.makedirs(dest_modality_dir, exist_ok=True)

    seq_path = os.path.join(source_dir, sessionID, modality+".npz")
    seq = np.load(seq_path)["data"]

    for start in range(0, len(seq), frame_step):
        end = min(start+frame_step, len(seq))
        chunk = seq[start:end]

        dest_path = os.path.join(dest_modality_dir, sessionID + "_" + str(start).zfill(6) + "_" + str(end).zfill(6) + ".npz")
        np.savez_compressed(dest_path, data=chunk)

if __name__ == "__main__":
    source_dir = "../no-chunked/"
    dest_dir = "../chunked/"

    nseconds = 5
    modalities = tqdm([
        ("audio_covarep", 100),
        ("audio_formant", 100),
        ("facial_2d_landmarks", 25),
        ("facial_3d_landmarks", 25),
        ("facial_aus", 25),
        ("facial_hog", 25),
        ("gaze", 25),
        ("head_pose", 25)
    ])

    sessionIDs = tqdm(sorted(os.listdir(source_dir)), leave=False)
    for modality, fps in modalities:
        frame_step = fps * nseconds
        for sessionID in sessionIDs:
            process_video(sessionID, modality)

