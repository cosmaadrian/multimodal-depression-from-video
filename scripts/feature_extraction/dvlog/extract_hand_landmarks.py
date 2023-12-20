import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from joblib import Parallel, delayed
from utils import load_video

def to_numpy_array(result):
   landmarks = []
   for res in result:
       landmarks.append([res.x, res.y, res.z, res.visibility, res.presence])
   return np.array(landmarks)

def create_hand_landmarker_object(hand_landmarker_path):
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path=hand_landmarker_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.VIDEO,
        num_hands = 2
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    return landmarker

def extract_landmarks_from_video(videoID):

    landmarker = create_hand_landmarker_object(args.hand_landmarker_path)

    video_path = os.path.join(args.video_dir, videoID+".mp4")
    frame_generator = load_video(video_path)

    os.makedirs(args.no_landmarks_output_dir, exist_ok=True)
    dst_no_landmarks_indeces_path = os.path.join(args.no_landmarks_output_dir, videoID+".npz")

    os.makedirs(args.landmarks_output_dir, exist_ok=True)
    dst_landmarks_path = os.path.join(args.landmarks_output_dir, videoID+".npz")

    if os.path.exists(dst_landmarks_path):
        print(f"Video {videoID} has already been processed.")

    else:
        frame_idx = 0
        no_landmarks_indeces = []
        hand_landmarks_seq = []
        while True:
            try:
                frame = frame_generator.__next__()
            except StopIteration:
                break

            print(f"\tProcessing frame {frame_idx} from video {videoID}", end="\r")

            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(mp_frame, frame_idx)

            if len(result.hand_landmarks) == 0:
                no_landmarks_indeces.append(frame_idx)
                left_hand_landmarks = np.zeros( (21,5) )
                right_hand_landmarks = np.zeros( (21,5) )

            elif len(result.hand_landmarks) == 1:

                if result.handedness[0][0].index == 0:
                    left_hand_landmarks = to_numpy_array(result.hand_landmarks[0])
                    right_hand_landmarks = np.zeros( (21,5) )
                else:
                    left_hand_landmarks = np.zeros( (21,5) )
                    right_hand_landmarks = to_numpy_array(result.hand_landmarks[0])

            else:
                left_hand_landmarks = to_numpy_array(result.hand_landmarks[0])
                right_hand_landmarks = to_numpy_array(result.hand_landmarks[1])

            hand_landmarks = np.stack((left_hand_landmarks, right_hand_landmarks))
            hand_landmarks_seq.append(hand_landmarks)

            frame_idx += 1

        np.savez_compressed(dst_no_landmarks_indeces_path, data=np.array(no_landmarks_indeces))
        np.savez_compressed(dst_landmarks_path, data=np.array(hand_landmarks_seq))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hand-landmarker-path", type=str, default="./features_extractors/hand_landmarker.task")
    parser.add_argument("--video-dir", type=str, default="./data/D-vlog/videos")
    parser.add_argument("--landmarks-output-dir", type=str, default="./data/D-vlog/data/hand_landmarks")
    parser.add_argument("--no-landmarks-output-dir", type=str, default="./data/D-vlog/data/no_hand_idxs")

    args = parser.parse_args()

    videoIDs = [sample.split(".")[0] for sample in sorted(os.listdir(args.video_dir))]

    loop = tqdm(videoIDs)
    Parallel(n_jobs=10, backend="multiprocessing", prefer="processes")(delayed(extract_landmarks_from_video)(video) for video in loop)
