import os
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    root_dir = "./data/"
    videoIDs = sorted(os.listdir(root_dir))

    train_split = pd.read_csv("./splits/training.csv", index_col=0)
    val_split = pd.read_csv("./splits/validation.csv", index_col=0)
    test_split = pd.read_csv("./splits/test.csv", index_col=0)
    dataset = pd.concat([train_split, val_split, test_split], ignore_index=True)

    for videoID in tqdm(videoIDs):
        # getting audio frame rate
        audio_frame_rate = dataset[dataset["video_id"] == videoID]["audio_frame_rate"].values[0]

        # getting the total number of frames that compose the video
        all_chunk_files = sorted(os.listdir(f'{root_dir}/{videoID}/audio_pase_embeddings/'))
        video_frame_length = int(all_chunk_files[-1].split(".")[0].split("_")[-1])

        # loading time slots where voice was detected
        voice_slots = np.load(f'{root_dir}/{videoID}/voice_activity.npz')['data']

        # creating mask
        no_voice_idxs = []
        for frame_idx in range(video_frame_length):
            frame_second = frame_idx / audio_frame_rate

            is_voice = False
            for voice_slot in voice_slots:
                if frame_second >= voice_slot[0] and frame_second <= voice_slot[1]:
                    is_voice = True

            if not is_voice:
                no_voice_idxs.append(frame_idx)

        dst_path = f"{root_dir}/{videoID}/no_voice_idxs.npz"
        np.savez_compressed(dst_path, data=np.array(no_voice_idxs))
