import os
import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":
    data_dir = sys.argv[1]
    split_path = sys.argv[2]
    dest_path = sys.argv[3]

    data_split_df = pd.read_csv(split_path, index_col=0)
    data_split = list(data_split_df.itertuples(index=False, name=None))

    new_split = []
    for videoID, label, gender, duration, channelID, _ in data_split:
        audio_embed_dir = os.path.join(data_dir, videoID, "audio_pase_embeddings")
        audio_embed_n_frames = int(sorted(os.listdir(audio_embed_dir))[-1].split(".")[0].split("_")[-1])
        audio_frame_rate = audio_embed_n_frames / duration

        video_embed_dir = os.path.join(data_dir, videoID, "face_emonet_embeddings")
        video_embed_n_frames = int(sorted(os.listdir(video_embed_dir))[-1].split(".")[0].split("_")[-1])
        video_frame_rate = video_embed_n_frames / duration

        new_split.append( (videoID, label, gender, duration, audio_embed_n_frames, audio_frame_rate, video_embed_n_frames, video_frame_rate, channelID) )

    new_df = pd.DataFrame(new_split, columns=["video_id", "label", "gender", "duration", "audio_nframes", "audio_frame_rate", "video_nframes", "video_frame_rate", "channel_id"])

    print(data_split_df)
    print(new_df)

    new_df.to_csv(dest_path)
