import os
import sys

# python3 -m pip install --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz
import yt_dlp

import argparse
import pandas as pd
from tqdm import tqdm

"""Example of use:

    python3 get_youtube_videos.py --csv-path ./databases/D-vlog/d-vlog.csv \
                                  --column-video-id video_id \
                                  --dest-dir ./databases/D-vlog/video/
                                  --dest-not-found ./databases/D-vlog/not_found.csv
"""

if __name__ == "__main__":
    # -- parsing command line arguments
    parser = argparse.ArgumentParser(description="Script to automatically downloading Youtube videos.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--csv-path", default="./databases/D-vlog/d-vlog.csv", type=str, help="CSV file where it is specified the Youtube video IDs.")
    parser.add_argument("--column-video-id", default="video_id", type=str, help="Name of the CSV's column where the Youtube video IDs are.")
    parser.add_argument("--dest-dir", default="./databases/D-vlog/videos/", type=str, help="Directory where the videos will be store.")
    parser.add_argument("--dest-not-found", default="./databases/D-vlog/not_found.csv", type=str, help="CSV file including the IDs of the videos that could not be download.")

    args = parser.parse_args()

    # -- creating directory if it does not exist
    os.makedirs(args.dest_dir, exist_ok=True)

    # -- obtaining the Youtube video IDs
    videoIDs = pd.read_csv(args.csv_path)[args.column_video_id].tolist()

    # -- downloading the Youtube videos with the best quality as possible
    not_found = []
    for videoID in tqdm(videoIDs):
        video_url = "https://www.youtube.com/watch?v=" + videoID
        dst_path = os.path.join(args.dest_dir, videoID + ".mp4")

        ydl_opts = {
            "quiet": False,
            "format": "mp4",
            "outtmpl": dst_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
            except yt_dlp.utils.DownloadError as e:
                if "private" in str(e.exc_info[1]).lower():
                    not_found.append((videoID, "private"))
                elif "unavailable" in str(e.exc_info[1]).lower():
                    not_found.append((videoID, "unavailable"))
                else:
                    not_found.append((videoID, "unknown"))

    # -- writing a CSV file for not-found video IDs
    not_found_df = pd.DataFrame(not_found, columns=[args.column_video_id, "exception_cause"])
    not_found_df.to_csv(args.dest_not_found)
