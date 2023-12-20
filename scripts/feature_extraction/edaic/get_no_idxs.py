import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=str, default="./data/E-DAIC/original_data/")
    parser.add_argument("--dest-dir", type=str, default="./data/E-DAIC/no-chunked/")
    args = parser.parse_args()

    sessionIDs = sorted( [x.split(".")[0] for x in os.listdir(args.data_dir)] )

    os.makedirs(os.path.join(args.dest_dir, "no_voice_idxs"), exist_ok = True)
    os.makedirs(os.path.join(args.dest_dir, "no_face_idxs"), exist_ok = True)

    modalities = ["voice", "face"]
    for sessionID in tqdm(sessionIDs):
        for modality in modalities:
            no_modality_path = os.path.join(args.dest_dir, "no_"+modality+"_idxs", sessionID+".npz")

            seq = np.array([], dtype=np.float32)
            np.savez_compressed(no_modality_path, data=seq)
