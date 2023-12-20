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

def load_face(image_path, target_width=256, target_height=256, transform=transforms.ToTensor()):
    image = np.load(image_path)["data"].astype(np.uint8)

    assert(image.ndim==3 and image.shape[2]==3)
    assert(image.dtype == np.uint8)

    image = cv2.resize(image, (target_width, target_height))
    image = np.ascontiguousarray(image)
    tensor_img = transform(image).to(args.cuda_device)

    return torch.unsqueeze(tensor_img, 0)

def process_video(videoID):
    video_dir = os.path.join(args.faces_dir, videoID)
    faceIDs = sorted(os.listdir(video_dir))

    dst_embedding_path = os.path.join(args.face_embeddings_output_dir, videoID+".npz")

    if os.path.exists(dst_embedding_path):
        print(f"Skipping sample {videoID} because it was already processed")
    else:
        embedding_seq = np.empty((0,256))
        for faceID in faceIDs:
            face_path = os.path.join(video_dir, faceID)
            face = load_face(face_path)

            emonet_embed, out = emonet(face)
            embedding_seq = np.vstack( (embedding_seq, emonet_embed.cpu().detach().numpy()) )

        np.savez_compressed(dst_embedding_path, data=embedding_seq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cpu")
    parser.add_argument("--n-expression", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default="./feature_extractors/emonet/pretrained/emonet_8.pth")
    parser.add_argument("--faces-dir", type=str, default="./data/D-vlog/data/faces/")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--left-index", type=int, default=0)
    parser.add_argument("--right-index", type=int, default=961)
    parser.add_argument("--face-embeddings-output-dir", required=True, type=str)
    args = parser.parse_args()

    emotion_map = {0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt', 8:'none'}

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
    emonet = EmoNet(n_expression=args.n_expression).to(args.cuda_device)
    emonet.load_state_dict(checkpoint)
    emonet.eval()

    os.makedirs(args.face_embeddings_output_dir, exist_ok=True)

    videoIDs = [video for video in sorted(os.listdir(args.faces_dir))][args.left_index:args.right_index]
    loop = tqdm(videoIDs)
    joblib.Parallel(n_jobs=10)(
        joblib.delayed(process_video)(videoID) for videoID in loop
    )
