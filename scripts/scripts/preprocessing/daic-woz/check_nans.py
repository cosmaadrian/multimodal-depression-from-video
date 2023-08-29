import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    root_dir = "./data/"
    sessionIDs = sorted( os.listdir(root_dir) )

    for sessionID in tqdm(sessionIDs):
        session_dir = os.path.join(root_dir, sessionID)
        modalityIDs = [m for m in sorted( os.listdir(session_dir) ) if ".npz" not in m]

        for modalityID in tqdm(modalityIDs, leave = False):
            modality_dir = os.path.join(session_dir, modalityID)
            chunkIDs = sorted( os.listdir(modality_dir) )

            for chunkID in tqdm(chunkIDs, leave = False):
                chunk_path = os.path.join(modality_dir, chunkID)
                chunk_data = np.load(chunk_path)["data"]

                if np.isnan(chunk_data).any():
                    print(chunk_path, np.isnan(chunk_data).any())
