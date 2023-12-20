import os
import cv2
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import load_video

def to_numpy_array(result):
   landmarks = []
   for res in result[0]:
       landmarks.append([res.x, res.y, res.z, res.visibility, res.presence])
   return np.array(landmarks)

def create_body_pose_landmarker_object(pose_landmarker_path):
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path=pose_landmarker_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    return landmarker

def process_video(videoID):
    landmarker = create_body_pose_landmarker_object(args.pose_landmarker_path)

    video_path = os.path.join(args.video_dir, videoID+".mp4")
    frame_generator = load_video(video_path)

    dst_no_landmarks_indeces_path = os.path.join(args.no_landmarks_output_dir, videoID+".npz")
    dst_landmarks_path = os.path.join(args.landmarks_output_dir, videoID+".npz")

    if os.path.exists(dst_landmarks_path):
        print(f"Video {videoID} has already been processed.")
    else:
        frame_idx = 0
        no_landmarks_indeces = []
        pose_landmarks_seq = []
        while True:
            try:
                frame = frame_generator.__next__()
            except StopIteration:
                break
            print(f"\tProcessing frame {frame_idx} from video {videoID}", end="\r")

            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(mp_frame, frame_idx)

            if len(result.pose_landmarks) == 0:
                no_landmarks_indeces.append(frame_idx)
                pose_landmarks = np.zeros( (33,5) )
            else:
                pose_landmarks = to_numpy_array(result.pose_landmarks)
            pose_landmarks_seq.append(pose_landmarks)

            frame_idx += 1

        np.savez_compressed(dst_no_landmarks_indeces_path, data=np.array(no_landmarks_indeces))
        np.savez_compressed(dst_landmarks_path, data=np.array(pose_landmarks_seq))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pose-landmarker-path", type=str, default="./feature_extractors/body_pose_landmarker.task")
    parser.add_argument("--video-dir", type=str, default="./data/D-vlog/videos/")
    parser.add_argument("--no-landmarks-output-dir", required=True, type=str)
    parser.add_argument("--landmarks-output-dir", required=True, type=str)
    args = parser.parse_args()

    os.makedirs(args.no_landmarks_output_dir, exist_ok=True)
    os.makedirs(args.landmarks_output_dir, exist_ok=True)

    videoIDs = [sample.split(".")[0] for sample in sorted(os.listdir(args.video_dir))]
    loop = tqdm(videoIDs)
    joblib.Parallel(n_jobs=8)(
        joblib.delayed(process_video)(videoID) for videoID in loop
    )
