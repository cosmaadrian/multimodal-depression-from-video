import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    root_dir = "./data/"
    modality = "audio_covarep"
    sessionIDs = sorted(os.listdir(root_dir))

    for sessionID in tqdm(sessionIDs):
        session_dir = os.path.join(root_dir, sessionID)
        modality_dir = os.path.join(session_dir, modality)
        chunkIDs = sorted( os.listdir(modality_dir) )

        for chunkID in tqdm(chunkIDs, leave = False):
            chunk_path = os.path.join(modality_dir, chunkID)

            data = np.load(chunk_path)["data"]
            if np.isneginf(data).any():
                # data[np.isneginf(data)] = 0.
                print(chunkID, data.shape, np.isneginf(data).any())
                # np.savez_compressed(chunk_path, data=data)
