import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--splits-dir", type=str, default="./data/D-vlog/splits/")
    parser.add_argument("--data-dir", type=str, default="./data/D-vlog/no-chunked/")
    parser.add_argument("--audio-embeddings-dir", type=str, default="audio_pase_embeddings")
    parser.add_argument("--audio-activity-dir", type=str, default="./data/D-vlog/no-chunked/audio-activity/")
    parser.add_argument("--dest-dir", type=str, default="no_voice_idxs")
    args = parser.parse_args()


    videoIDs = sorted(os.listdir(args.data_root_dir))

    train_split = pd.read_csv(f"{args.splits_dir}/training.csv", index_col=0)
    val_split = pd.read_csv(f"{splits_dir}/validation.csv", index_col=0)
    test_split = pd.read_csv(f"{splits_dir}/test.csv", index_col=0)
    dataset = pd.concat([train_split, val_split, test_split], ignore_index=True)

    os.makedirs(f"{args.data_dir}/no_voice_idxs/", exist_ok = True)
    for videoID in tqdm(videoIDs):
        audio_frame_rate = dataset[dataset["video_id"] == videoID]["audio_frame_rate"].values[0]

        audio_path = os.path.join(args.data_dir, args.audio_embeddings_dir, videoID+".npz")
        audio_frame_length = np.load(audio_path)['data'].shape[0]

        voice_slots = np.load(f'{args.voice_activity_dir}/{videoID}.npz')["data"]

        no_voice_idxs = []
        for frame_idx in range(video_frame_length):
            frame_second = frame_idx / audio_frame_rate

            is_voice = False
            for voice_slot in voice_slots:
                if frame_second >= voice_slot[0] and frame_second <= voice_slot[1]:
                    is_voice = True

            if not is_voice:
                no_voice_idxs.append(frame_idx)

        dst_path = f"{args.data_dir}/{args.dest_dir}/{videoID}.npz"
        np.savez_compressed(dest_path, data=np.array(no_voice_idxs))
