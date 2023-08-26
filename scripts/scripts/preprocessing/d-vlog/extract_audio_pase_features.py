import os
import torch
import argparse
import warnings
import torchaudio
import numpy as np
from tqdm import tqdm

from pase.models.frontend import wf_builder

def process_audio(audioID):
    audio_path = os.path.join(args.audio_dir, audioID+".wav")
    waveform, speech_rate = torchaudio.load(audio_path)

    # -- creating destination path
    dst_embedding_path = os.path.join(args.audio_embeddings_output_dir, audioID+".npz")
    if os.path.exists(dst_embedding_path):
        print(f"Skipping sample {audioID} because it was already processed")
    else:
        pase_seq = np.empty((0,256))
        step = speech_rate * args.chunk_seconds
        waveform = waveform.unsqueeze(0).to(args.cuda_device)
        waveform_length = waveform.shape[-1]
        for t in range(0, waveform_length, step):
            input_chunk = waveform[:, :, t:t+step]
            if input_chunk.shape[-1] > 320:
                pase_out = pase(input_chunk).squeeze(0)
                pase_emb = torch.transpose(pase_out, 0, 1)
                # -- appending the processed chunk
                pase_seq = np.vstack((pase_seq, pase_emb.cpu().detach().numpy()))

        # -- saving PASE+ audio embeddings
        np.savez_compressed(dst_embedding_path, data=pase_seq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computing PASE+ audio embeddings.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda:0", help="Choose a GPU device")
    parser.add_argument("--config-file", type=str, default="./tools/pase/cfg/frontend/PASE+.cfg", help="PASE+ encoder configuration file")
    parser.add_argument("--checkpoint", type=str, default="./tools/pase/FE_e199.ckpt", help="PASE+ encoder pre-trained checkpoint file")
    parser.add_argument("--audio-dir", type=str, default="./databases/D-vlog/data/audios/", help="Root directory where Youtube audios are stored")
    parser.add_argument("--chunk-seconds", type=int, default=5, help="Number of seconds for the audio chunk that split the large audio clips")
    parser.add_argument("--left-index", type=int, default=0, help="Position index from where to start to process videos")
    parser.add_argument("--right-index", type=int, default=861, help="Position index where to finish to process videos")
    parser.add_argument("--audio-embeddings-output-dir", required=True, type=str, help="Directory where to save the computed 256-dimensional audio embeddings for each audio that compose the database")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    # -- building and pre-training the PASE+ encoder
    pase = wf_builder(args.config_file).to(args.cuda_device).eval()
    pase.load_pretrained(args.checkpoint, load_last=True, verbose=False)

    # -- creating output directories
    os.makedirs(args.audio_embeddings_output_dir, exist_ok=True)

    # -- processing the database
    audioIDs = [sample.split(".")[0] for sample in sorted(os.listdir(args.audio_dir))][args.left_index:args.right_index]
    for audioID in tqdm(audioIDs):
        if audioID == "2jyaHz_O-co":
            process_audio(audioID)
