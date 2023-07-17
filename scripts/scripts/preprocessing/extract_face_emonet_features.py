import os
import cv2
import torch
import joblib
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms

import sys
sys.path.append('./')
from tools.emonet.emonet.models.emonet import EmoNet

"""
How post-processing the EmoNet's output w.r.t. the expression classification scores, the valence, and the arousal

Author comment on a Github issue:

   'The output of the network for the expression values are the score before softmax for each emotion. This is the reason why the values can be negative. If you want to transform this vector into a probability distribution please apply a softmax.'

So, after inspecting the "emonet/emonet/evaluation.py" script:

expression_pred = np.argmax(np.squeeze(out["expression"].cpu().numpy()), axis=1) # Obtaining most probable emotional class reflected on the face
valence_pred = np.squeeze(out["valence"].cpu().numpy())
arousal_pred = np.squeeze(out["arousal"].cpu().numpy())

"""

def load_face(image_path, target_width=256, target_height=256, transform=transforms.ToTensor()):
    # -- loading RGB image
    image = np.load(image_path)["data"].astype(np.uint8)

    # -- sanity checkings
    assert(image.ndim==3 and image.shape[2]==3)
    assert(image.dtype == np.uint8)

    # -- EmoNet implementation details
    image = cv2.resize(image, (target_width, target_height))
    image = np.ascontiguousarray(image)
    tensor_img = transform(image).to(args.cuda_device)

    return torch.unsqueeze(tensor_img, 0)

def process_video(videoID):
    video_dir = os.path.join(args.faces_dir, videoID)
    faceIDs = sorted(os.listdir(video_dir))

    # -- creating destination path
    dst_embedding_path = os.path.join(args.face_embeddings_output_dir, videoID+".npz")
    dst_emotion_path = os.path.join(args.emotion_stats_output_dir, videoID+".pkl")

    if os.path.exists(dst_embedding_path):
        print(f"Skipping sample {videoID} because it was already processed")
    else:
        embedding_seq = np.empty((0,256))
        emonet_stats = {"heatmap": np.empty( (0, 68, 64, 64) ), "expression": np.empty( (0,8) ), "valence": np.empty( (0,) ), "arousal": np.empty( (0,) )}
        for faceID in faceIDs:
            face_path = os.path.join(video_dir, faceID)
            face = load_face(face_path)

            # -- EmoNet forward pass -> {"heatmap", "expression", "valence", "arousal"}
            emonet_embed, out = emonet(face)
            embedding_seq = np.vstack( (embedding_seq, emonet_embed.cpu().detach().numpy()) )

            emonet_stats["heatmap"] = np.vstack( (emonet_stats["heatmap"], out["heatmap"].cpu().detach().numpy()) )
            emonet_stats["expression"] = np.vstack( (emonet_stats["expression"], out["expression"].cpu().detach().numpy()) )
            emonet_stats["valence"] = np.hstack( (emonet_stats["valence"], out["valence"].cpu().detach().numpy()) )
            emonet_stats["arousal"] = np.hstack( (emonet_stats["arousal"], out["arousal"].cpu().detach().numpy()) )

        # -- saving EmoNet face embeddings + emotion, valence, and arousal values
        np.savez_compressed(dst_embedding_path, data=embedding_seq)
        with open(dst_emotion_path, "wb") as handle:
            pickle.dump(emonet_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # -- how to read/load these kind of pickle files?
        # with open(path_to_pickle, 'rb') as handle:
        #     loaded_emonet_stats = pickle.load(handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecting faces and facial landmarks from videos.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cpu", help="Choose a GPU device")
    parser.add_argument("--n-expression", type=int, default=8, help="Number of emotional classes")
    parser.add_argument("--checkpoint", type=str, default="./tools/emonet/pretrained/emonet_8.pth", help="EmoNet encoder pre-trained checkpoint file")
    parser.add_argument("--faces-dir", type=str, default="./databases/D-vlog/data/faces/", help="Root directory where Youtube audios are stored")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of faces to process in parallel")
    parser.add_argument("--left-index", type=int, default=0, help="Position index from where to start to process videos")
    parser.add_argument("--right-index", type=int, default=861, help="Position index where to finish to process videos")
    parser.add_argument("--face-embeddings-output-dir", required=True, type=str, help="Directory where to save the computed 256-dimensional face embeddings for each video that compose the database")
    parser.add_argument("--emotion-stats-output-dir", required=True, type=str, help="Directory where to save the computed emotion statistics for each face of each video that compose the database")
    args = parser.parse_args()

    # -- label implementation details
    emotion_map = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'}

    # -- building and pre-training the EmoNet encoder
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
    emonet = EmoNet(n_expression=args.n_expression).to(args.cuda_device)
    emonet.load_state_dict(checkpoint)
    emonet.eval()

    # -- creating output directories
    os.makedirs(args.face_embeddings_output_dir, exist_ok=True)
    os.makedirs(args.emotion_stats_output_dir, exist_ok=True)

    # -- processing the database
    videoIDs = [video for video in sorted(os.listdir(args.faces_dir))][args.left_index:args.right_index]
    loop = tqdm(videoIDs)
    joblib.Parallel(n_jobs=10)(
        joblib.delayed(process_video)(videoID) for videoID in loop
    )
