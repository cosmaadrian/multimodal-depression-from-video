from pyannote.audio import Pipeline
import numpy as np
import glob
from tqdm import tqdm
import torch
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--pretrained-model", type=str, default="pyannote/voice-activity-detection")
    parser.add_argument("--audio-dir", type=str, default="./data/D-vlog/wavs/")
    parser.add_argument("--dest-dir", type=str, default="./data/D-vlog/no-chunked/audio_activity/")
    parser.add_argument("--use-auth-token", required=True, type=str)
    args = parser.parse_args()


    device = torch.device(args.cuda_device)
    pipeline = Pipeline.from_pretrained(args.pretrained_model, use_auth_token=args.use_auth_token)
    pipeline.to(device)

    audio_ids = sorted( [id.split(".")[0] for id in os.listdir(args.audio_dir)] )

    for audio_id in tqdm(audio_ids):
        audio_activity = []
        audio_file = os.path.join(args.audio_dir, audio_id+".wav")
        output = pipeline(audio_file)

        for turn, _, speaker in output.itertracks(yield_label=True):
            audio_activity.append([float(turn.start), float(turn.end)])

        audio_activity = np.array(audio_activity)

        os.makedirs(args.dest_dir, exist_ok = True)
        dst_path = os.path.join(args.dest_dir, audio_id+".npz")
        np.savez_compressed(dst_path, data=audio_activity)
