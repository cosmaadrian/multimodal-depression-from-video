import os
import torch
import pathlib
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import load_video, get_video_info

import sys

def extract_gaze_from_video(videoID, config):
    gaze_estimator = GazeEstimator(config)

    video_path = os.path.join(args.video_dir, videoID+".mp4")
    frame_generator = load_video(video_path)

    fps, frame_count, frame_width, frame_height = get_video_info(video_path)

    os.makedirs(args.no_gaze_output_dir, exist_ok=True)
    dst_no_gaze_indeces_path = os.path.join(args.no_gaze_output_dir, videoID+".npz")

    os.makedirs(args.landmarks_output_dir, exist_ok=True)
    dst_landmarks_path = os.path.join(args.landmarks_output_dir, videoID+".npz")

    if os.path.exists(dst_landmarks_path):
        print(f"Video {videoID} has already been processed.")
        return

    no_gaze_indexes = []
    gaze_seq = []
    length = config.demo.gaze_visualization_length
    for frame_idx, frame in enumerate(frame_generator):
        undistorted = cv2.undistort(frame, gaze_estimator.camera.camera_matrix, gaze_estimator.camera.dist_coefficients)
        faces = gaze_estimator.detect_faces(undistorted)

        for j, face in enumerate(faces):
            gaze_estimator.estimate_gaze(undistorted, face)
            gaze_seq.append(face.gaze_vector)
            break

        if len(faces) and (frame_idx + 1) % 1024 == 0:
            print(f"Video {videoID} - frame {frame_idx}/{frame_count}")

        if not len(faces):
            no_gaze_indexes.append(frame_idx)
            gaze_seq.append(np.zeros(3, dtype=np.float32))

    np.savez_compressed(dst_no_gaze_indeces_path, data=np.array(no_gaze_indexes))
    np.savez_compressed(dst_landmarks_path, data=np.array(gaze_seq))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video-dir", type=str, default="./data/D-vlog/videos")
    parser.add_argument("--gaze-output-dir", default="./data/D-vlog/data/gaze_features", type=str)
    parser.add_argument("--no-gaze-output-dir", default="./data/D-vlog/data/no_gaze_idxs/", type=str)
    parser.add_argument("--mode", default="eth-xgaze", type=str)
    parser.add_argument("--mpiigaze-dir", required=True, type=str)
    parser.add_argument("--eth-xgaze-config-path", required=True, type=str)
    parser.add_argument('--face-detector', default='mediapipe', type=str)
    parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'])
    args = parser.parse_args()

    sys.path.append(args.mpiigaze_dir)
    from ptgaze.utils import download_ethxgaze_model, get_3d_face_model, download_dlib_pretrained_model
    from ptgaze.gaze_estimator import GazeEstimator
    from ptgaze.common import Face, FacePartsName, Visualizer
    from omegaconf import DictConfig, OmegaConf

    config = OmegaConf.load(args.eth_xgaze_config_path)

    if args.face_detector:
        config.face_detector.mode = args.face_detector

    if args.device:
        config.device = args.device

    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        print('Run on CPU because CUDA is not available.')

    if config.face_detector.mode == 'dlib':
        download_dlib_pretrained_model()

    package_root = pathlib.Path(args.mpiigaze_dir+'/ptgaze').resolve()
    config.PACKAGE_ROOT = package_root.as_posix()

    download_ethxgaze_model()

    videoIDs = [sample.split(".")[0] for sample in sorted(os.listdir(args.video_dir))]

    loop = tqdm(videoIDs)

    Parallel(n_jobs=6)(delayed(extract_gaze_from_video)(video, config) for video in loop)

