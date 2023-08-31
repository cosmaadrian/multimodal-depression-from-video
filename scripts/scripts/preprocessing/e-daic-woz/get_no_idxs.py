import os
import numpy as np
from tqdm import tqdm

def compute_nframes(session_path, modality):
    session_modality_path = os.path.join(session_path, modality)
    chunks = sorted( os.listdir(session_modality_path) )
    nframes = int(chunks[-1].split(".")[0].split("_")[-1])

    return nframes

if __name__ == "__main__":
    data_dir = "./data/"
    sessionIDs = sorted( os.listdir(data_dir) )

    modalities = ["voice", "face"]
    for sessionID in tqdm(sessionIDs):
        session_path = os.path.join(data_dir, sessionID)

        for modality in modalities:
            no_modality_path = os.path.join(session_path, "no_"+modality+"_idxs.npz")

            # nframes = compute_nframes(session_path, modality)
            seq = np.array([], dtype=np.float32)
            np.savez_compressed(no_modality_path, data=seq)
