import os
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor

from utils import *

def process_video(videoID):
    video_path = os.path.join(args.video_dir, videoID+".mp4")
    frame_generator = load_video(video_path)

    dst_no_face_indeces_path = os.path.join(args.no_faces_output_dir, videoID+".npz")
    dst_landmarks_path = os.path.join(args.landmarks_output_dir, videoID+".npz")
    dst_faces_path = os.path.join(args.faces_output_dir, videoID+".npz")

    if os.path.exists(dst_landmarks_path):
        print(f"Video {videoID} has already been processed.")
    else:
        frame_idx = 0
        no_face_indeces = []
        landmark_sequence = []
        face_sequence = []
        while True:
            try:
                frame = frame_generator.__next__()
            except StopIteration:
                break

            print(f"\tProcessing frame {frame_idx} from video {videoID}", end="\r")
            detected_faces = face_detector(frame, rgb=False)

            if len(detected_faces) == 0:
                no_face_indeces.append(frame_idx)
                landmark_sequence.append( np.zeros((68, 3)) )
                face_sequence.append( np.zeros((128, 128, 3)) )
            else:
                landmarks, scores = landmark_detector(frame, detected_faces, rgb=False)
                scores = np.moveaxis(scores, 0, -1)
                score_landmarks = np.hstack((landmarks[0], scores[:, :1]))
                landmark_sequence.append(score_landmarks)

                transformed_frame, transformed_landmarks = affine_transform(
                    frame,
                    landmarks[0],
                    np.load(args.mean_face_path),
                    grayscale=False,
                )

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

            frame_idx += 1

        np.savez_compressed(dst_no_face_indeces_path, data=np.array(no_face_indeces))
        np.savez_compressed(dst_landmarks_path, data=np.array(landmark_sequence))
        np.savez_compressed(dst_faces_path, data=np.array(face_sequence))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecting faces and facial landmarks from videos.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--video-dir", type=str, default="./databases/D-vlog/data/videos/")
    parser.add_argument("--left-index", type=int, default=0)
    parser.add_argument("--right-index", type=int, default=961)
    parser.add_argument("--mean-face-path", type=str, default="./dvlog/20words_mean_face.npy")
    parser.add_argument("--no-faces-output-dir", required=True, type=str)
    parser.add_argument("--landmarks-output-dir", required=True, type=str)
    parser.add_argument("--faces-output-dir", required=True, type=str)
    args = parser.parse_args()

    face_detector = RetinaFacePredictor(
        threshold=0.8, device=args.cuda_device,
        model=RetinaFacePredictor.get_model('resnet50')
    )

    landmark_detector = FANPredictor(
        device=args.cuda_device, model=FANPredictor.get_model('2dfan2_alt')
    )

    os.makedirs(args.no_faces_output_dir, exist_ok=True)
    os.makedirs(args.landmarks_output_dir, exist_ok=True)
    os.makedirs(args.faces_output_dir, exist_ok=True)

    videoIDs = [sample.split(".")[0] for sample in sorted(os.listdir(args.video_dir))][args.left_index:args.right_index]
    for videoID in tqdm(videoIDs):
        process_video(videoID)
