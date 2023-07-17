import os
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# https://github.com/hhj1897/face_detection
from ibug.face_detection import RetinaFacePredictor
# https://github.com/hhj1897/face_alignment
from ibug.face_alignment import FANPredictor

from utils import *

def process_video(videoID):
    video_path = os.path.join(args.video_dir, videoID+".mp4")
    frame_generator = load_video(video_path)

    # -- creating destination paths
    dst_no_face_indeces_path = os.path.join(args.no_faces_output_dir, videoID+".npz")
    dst_landmarks_path = os.path.join(args.landmarks_output_dir, videoID+".npz")
    dst_faces_path = os.path.join(args.faces_output_dir, videoID+".npz")

    if os.path.exists(dst_landmarks_path):
        print(f"Video {videoID} has already been processed.")
    else:
        # -- processing video
        frame_idx = 0
        no_face_indeces = []
        landmark_sequence = []
        face_sequence = []
        while True:
            try:
                frame = frame_generator.__next__() ## -- BGR
            except StopIteration:
                break

            print(f"\tProcessing frame {frame_idx} from video {videoID}", end="\r")
            # -- -- detecting face
            detected_faces = face_detector(frame, rgb=False)

            # -- -- if there is no visible face
            if len(detected_faces) == 0:
                no_face_indeces.append(frame_idx)
                landmark_sequence.append( np.zeros((68, 3)) )
                face_sequence.append( np.zeros((128, 128, 3)) )
            # -- -- if a face has been detected
            else:
                # -- -- computing 68 facial landmarks
                landmarks, scores = landmark_detector(frame, detected_faces, rgb=False)
                # -- -- stacking scores for each landmark
                scores = np.moveaxis(scores, 0, -1)
                score_landmarks = np.hstack((landmarks[0], scores[:, :1]))
                landmark_sequence.append(score_landmarks)

                # -- -- applying affine transform to head-pose normalisation
                transformed_frame, transformed_landmarks = affine_transform(
                    frame,
                    landmarks[0],
                    np.load("./scripts/pre-processing/20words_mean_face.npy"),
                    grayscale=False,
                )

                # -- -- extraction region of interest
                start_idx = 0; stop_idx = 68
                crop_height = 128; crop_width = 128

                face_sequence.append(
                    cut_patch(
                        transformed_frame,
                        transformed_landmarks[start_idx:stop_idx],
                        crop_height//2,
                        crop_width//2,
                    ),
                )

            # -- updating to current frame index
            frame_idx += 1

        # -- saving detected faces and landmarks from the video
        np.savez_compressed(dst_no_face_indeces_path, data=np.array(no_face_indeces))
        np.savez_compressed(dst_landmarks_path, data=np.array(landmark_sequence))
        np.savez_compressed(dst_faces_path, data=np.array(face_sequence))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecting faces and facial landmarks from videos.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda:0", help="Choose a GPU device")
    parser.add_argument("--video-dir", type=str, default="./databases/D-vlog/data/videos/", help="Root directory where Youtube videos are stored")
    parser.add_argument("--left-index", type=int, default=0, help="Position index from where to start to process videos")
    parser.add_argument("--right-index", type=int, default=861, help="Position index where to finish to process videos")
    parser.add_argument("--no-faces-output-dir", required=True, type=str, help="Directory where to save an array indicating the frames where no face was found")
    parser.add_argument("--landmarks-output-dir", required=True, type=str, help="Directory where to save the computed landmarks for each video that compose the database")
    parser.add_argument("--faces-output-dir", required=True, type=str, help="Directory where to save the computed 128x128 facer for each video that compose the database")
    args = parser.parse_args()

    # -- building face detector
    face_detector = RetinaFacePredictor(
        threshold=0.8, device=args.cuda_device,
        model=RetinaFacePredictor.get_model('resnet50')
    )

    # -- building face aligner or landmark detector
    landmark_detector = FANPredictor(
        device=args.cuda_device, model=FANPredictor.get_model('2dfan2_alt')
    )

    # -- creating output directories
    os.makedirs(args.no_faces_output_dir, exist_ok=True)
    os.makedirs(args.landmarks_output_dir, exist_ok=True)
    os.makedirs(args.faces_output_dir, exist_ok=True)

    # -- loading database
    videoIDs = [sample.split(".")[0] for sample in sorted(os.listdir(args.video_dir))][args.left_index:args.right_index]
    for videoID in tqdm(videoIDs):
        process_video(videoID)
    # joblib.Parallel(n_jobs=6)(
    #     joblib.delayed(process_video)(videoID) for videoID in videoIDs
    # )

    """How to read/load this type of files?

        import numpy as np
        landmarks = np.load("./landmarks.npz")["data"]
        landmarks.shape # (397, 68, 2) -- (time, landmarks, coordinates)

        faces = np.load("./faces.npz")["data"]
        faces.shape # (397, 128, 128, 3) -- (time, height, width, channels)
    """
