import os
import sys

import argparse
import subprocess
import pandas as pd
from tqdm import tqdm

"""Example of use:

    python3 compute_wavs.py --csv-path ./databases/D-vlog/splits/train.csv \
                            --column-video-id videoID \
                            --video-dir ./videos/ \
                            --dest-dir ./data/D-vlog/audios/
"""

if __name__ == "__main__":
    # -- parsing command line arguments
    parser = argparse.ArgumentParser(description="Script to automatically downloading Youtube videos.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--csv-path", default="./databases/D-vlog/d-vlog.csv", type=str, help="CSV file where it is specified the Youtube video IDs.")
    parser.add_argument("--column-video-id", default="video_id", type=str, help="Name of the CSV's column where the Youtube video IDs are.")
    parser.add_argument("--video-dir", default="./databases/D-vlog/data/videos/", type=str, help="Directory where the videos were store.")
    parser.add_argument("--dest-dir", default="./databases/D-vlog/data/audios/", type=str, help="Directory where the WAVs will be store.")

    args = parser.parse_args()

    # -- creating directory if it does not exist
    os.makedirs(args.dest_dir, exist_ok=True)

    # -- obtaining the Youtube video IDs
    videoIDs = pd.read_csv(args.csv_path)[args.column_video_id].tolist()

    # -- extracting 16kHz mono-channel 16-bit waveforms from MP4 videos
    for videoID in tqdm(videoIDs[600:]):
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
