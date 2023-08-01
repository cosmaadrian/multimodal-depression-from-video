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
sys.path.append('/home/banamar/pytorch_mpiigaze_demo/')
from ptgaze.utils import download_ethxgaze_model, get_3d_face_model, download_dlib_pretrained_model
from ptgaze.gaze_estimator import GazeEstimator
from ptgaze.common import Face, FacePartsName, Visualizer
from omegaconf import DictConfig, OmegaConf

def extract_gaze_from_video(videoID, config, gaze_estimator):
    # -- creating video frame generator
    video_path = os.path.join(args.video_dir, videoID+".mp4")
    frame_generator = load_video(video_path)

    fps, frame_count, frame_width, frame_height = get_video_info(video_path)

    # -- creating destination directories
    os.makedirs(args.no_gaze_output_dir, exist_ok=True)
    dst_no_gaze_indeces_path = os.path.join(args.no_gaze_output_dir, videoID+".npz")

    os.makedirs(args.landmarks_output_dir, exist_ok=True)
    dst_landmarks_path = os.path.join(args.landmarks_output_dir, videoID+".npz")

    if os.path.exists(dst_landmarks_path):
        print(f"Video {videoID} has already been processed.")
        return

    # -- processing video
    no_gaze_indexes = []
    gaze_seq = []
    length = config.demo.gaze_visualization_length
    for frame_idx, frame in enumerate(frame_generator):
        undistorted = cv2.undistort(frame, gaze_estimator.camera.camera_matrix, gaze_estimator.camera.dist_coefficients)
        faces = gaze_estimator.detect_faces(undistorted)

        for j, face in enumerate(faces):
            gaze_estimator.estimate_gaze(undistorted, face)
            gaze_seq.append(face.gaze_vector)
            break # assume only one face per frame            

        if not len(faces):
            no_gaze_indexes.append(frame_idx)
            gaze_seq.append(np.zeros(3, dtype=np.float32))
        
        # if (frame_idx + 1) % 100 == 0:
        #     print(f"Video {videoID} - frame {frame_idx}/{frame_count}")
        #     break

    #  saving computed gaze from the video
    np.savez_compressed(dst_no_gaze_indeces_path, data=np.array(no_gaze_indexes))
    np.savez_compressed(dst_landmarks_path, data=np.array(gaze_seq))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimating gaze from videos.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video-dir", type=str, default="/home/banamar/perceiving-depression/databases/D-vlog/data/videos", help="Root directory where Youtube videos are stored")
    parser.add_argument("--landmarks-output-dir", default="/home/banamar/perceiving-depression/databases/D-vlog/data/gaze_features", type=str, help="Directory where to save the computed gaze for each video that compose the database")
    parser.add_argument("--no-gaze-output-dir", default="/home/banamar/perceiving-depression/databases/D-vlog/data/no_gaze", type=str, help="Directory where to save the frames where no gaze were found")
    parser.add_argument("--mode", default="eth-xgaze", type=str, help="Mode to use for gaze estimation")
    parser.add_argument("--eth_xgaze_config_path", default="/home/banamar/pytorch_mpiigaze_demo/ptgaze/data/configs/eth-xgaze.yaml", type=str, help="Path to config file for eth-xgaze")
    parser.add_argument('--face-detector', default='dlib', type=str, help='The method used to detect faces and find face landmarks')
    parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'], help='Device used for model inference.')
    parser.add_argument('--no-screen', action='store_true', help='If specified, the video is not displayed on screen, and saved to the output directory.')
    args = parser.parse_args()

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

    package_root = pathlib.Path('/home/banamar/pytorch_mpiigaze_demo/ptgaze').resolve()
    config.PACKAGE_ROOT = package_root.as_posix()

    # download_mpiifacegaze_model()
    download_ethxgaze_model()
    gaze_estimator = GazeEstimator(config)

    # -- loading database
    videoIDs = [sample.split(".")[0] for sample in sorted(os.listdir(args.video_dir))]

    loop = tqdm(videoIDs)

    Parallel(n_jobs=11)(delayed(extract_gaze_from_video)(video, config, gaze_estimator) for video in loop)

