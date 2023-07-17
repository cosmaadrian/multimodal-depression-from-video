from pyannote.audio import Pipeline
import numpy as np
import glob
import tqdm
import torch
device = torch.device('cuda:0')


pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token="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")
pipeline.to(device)

audio_files = sorted(glob.glob("/home/banamar/perceiving-depression/databases/D-vlog/data/audios/*.wav"))
processed_files = sorted(glob.glob("/home/banamar/perceiving-depression/databases/D-vlog/data/audio_activity/*.npz"))

for audio_file in tqdm.tqdm(audio_files):
    if audio_file.replace("audios", "audio_activity").replace(".wav", ".npz") in processed_files:
        print(f"Audio {audio_file} has already been processed.")
        continue

    audio_activity = []

    output = pipeline(audio_file)

    for turn, _, speaker in output.itertracks(yield_label=True):
        audio_activity.append([float(turn.start), float(turn.end)])
        # print(f"start={turn.start}s stop={turn.end}s")


    audio_activity = np.array(audio_activity)
    np.savez_compressed(audio_file.replace("audios", "audio_activity").replace(".wav", ".npz"), data=audio_activity)
    
