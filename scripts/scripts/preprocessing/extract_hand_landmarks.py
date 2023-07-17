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

def draw_landmarks_on_image_hands(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

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

    # -- creating pose landmark estimator
    landmarker = create_hand_landmarker_object(args.hand_landmarker_path)

    # -- creating video frame generator
    video_path = os.path.join(args.video_dir, videoID+".mp4")
    frame_generator = load_video(video_path)

    # -- creating destination directories
    os.makedirs(args.no_landmarks_output_dir, exist_ok=True)
    dst_no_landmarks_indeces_path = os.path.join(args.no_landmarks_output_dir, videoID+".npz")

    os.makedirs(args.landmarks_output_dir, exist_ok=True)
    dst_landmarks_path = os.path.join(args.landmarks_output_dir, videoID+".npz")

    if os.path.exists(dst_landmarks_path):
        print(f"Video {videoID} has already been processed.")

    else:
        # -- processing video
        frame_idx = 0
        no_landmarks_indeces = []
        hand_landmarks_seq = []
        while True:
            try:
                frame = frame_generator.__next__() ## -- BGR
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

            # -- saving annotated images to see if it is working
            # print('saving annotated images')
            # os.makedirs(f"images/{videoID}", exist_ok=True)
            # annotated_frame = draw_landmarks_on_image_hands(mp_frame.numpy_view(), result)
            # cv2.imwrite(f"images/{videoID}/{str(frame_idx).zfill(3)}.png", annotated_frame)

            frame_idx += 1

        # -- saving computed body pose landmarks from the video
        np.savez_compressed(dst_no_landmarks_indeces_path, data=np.array(no_landmarks_indeces))
        np.savez_compressed(dst_landmarks_path, data=np.array(hand_landmarks_seq))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimating body pose landmarks from videos.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hand-landmarker-path", type=str, default="./tools/landmarkers/hand_landmarker.task", help="Path to the hand landmarker task")
    parser.add_argument("--video-dir", type=str, default="./databases/D-vlog/data/videos", help="Root directory where Youtube videos are stored")
    parser.add_argument("--landmarks-output-dir", default="/home/banamar/perceiving-depression/databases/D-vlog/data/hand_landmarks", type=str, help="Directory where to save the computed body pose landmarks for each video that compose the database")
    parser.add_argument("--no-landmarks-output-dir", default="/home/banamar/perceiving-depression/databases/D-vlog/data/no_hand_landmarks", type=str, help="Directory where to save the frames where no body pose landmarks were found")

    args = parser.parse_args()

    # -- loading database
    videoIDs = [sample.split(".")[0] for sample in sorted(os.listdir(args.video_dir))]

    loop = tqdm(videoIDs)
    Parallel(n_jobs=10, backend="multiprocessing", prefer="processes")(delayed(extract_landmarks_from_video)(video) for video in loop)
