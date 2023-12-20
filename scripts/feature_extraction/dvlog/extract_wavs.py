import os
import sys

import argparse
import subprocess
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to automatically downloading Youtube videos.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--csv-path", default="./data/D-vlog/dvlog_video_ids.csv", type=str, help="CSV file where it is specified the video IDs.")
    parser.add_argument("--column-video-id", default="video_id", type=str, help="Name of the CSV's column where the Youtube video IDs are.")
    parser.add_argument("--video-dir", default="./data/D-vlog/data/videos/", type=str, help="Directory where the videos were store.")
    parser.add_argument("--dest-dir", default="./data/D-vlog/data/wavs/", type=str, help="Directory where the WAVs will be store.")

    args = parser.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)

    videoIDs = pd.read_csv(args.csv_path)[args.column_video_id].tolist()

    for videoID in tqdm(videoIDs):
        dst_path = os.path.join(args.dest_dir, videoID + ".wav")
        video_path = os.path.join(args.video_dir, videoID + ".mp4")

        subprocess.call(
            [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-ar", "16000",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                dst_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
