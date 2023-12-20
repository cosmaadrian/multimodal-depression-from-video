import os
import sys
import joblib
import numpy as np
from tqdm import tqdm

def process_video(videoID):
    dest_modality_dir = os.path.join(args.dest_dir, videoID.replace(".npz", ""), modality)
    os.makedirs(dest_modality_dir, exist_ok=True)

    seq_path = os.path.join(modality_dir, videoID)
    seq = np.load(seq_path)["data"]

    for start in range(0, len(seq), frame_step):
        end = min(start+frame_step, len(seq))
        chunk = seq[start:end]

        dest_path = os.path.join(dest_modality_dir, videoID.replace(".npz", "_" + str(start).zfill(6) + "_" + str(end).zfill(6) + ".npz"))
        np.savez_compressed(dest_path, data=chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-dir", type=str, default="./data/D-vlog/no-chunked/")
    parser.add_argument("--dest-dir", type=str, default="./data/D-vlog/data/")
    parser.add_argument("--nseconds", type=int, default=5)
    parser.add_argument("--modality-id", required=True, type=str)
    parser.add_argument("--no-idxs-id", required=True, type=str)
    parser.add_argument("--frame-rate", required=True, type=int)
    args = parser.parse_args()

    modalities = [(args.modality_id, args.frame_rate)]
    no_idxs_modalities = [args.no_idxs_id]

    for modality, fps in tqdm(modalities):
        frame_step = fps * args.nseconds
        modality_dir = os.path.join(args.source_dir, modality)

        videoIDs = tqdm(sorted(os.listdir(modality_dir)), leave=False)

        joblib.Parallel(n_jobs=8)(
            joblib.delayed(process_video)(videoID) for videoID in videoIDs
        )

    for no_idxs_folder in no_idxs_modalities:
         no_idxs_folder_path = os.path.join(args.source_dir, no_idxs_folder)
         no_idxs_videoIDs = tqdm(sorted(os.listdir(no_idxs_folder_path)), leave=False)

         for videoID in no_idxs_videoIDs:
             dest_modality_dir = os.path.join(args.dest_dir, videoID.replace(".npz", ""), no_idxs_folder)
             dest_path = dest_modality_dir + ".npz"
             np.savez_compressed(dest_path, data=np.load(os.path.join(no_idxs_folder_path, videoID))["data"])


