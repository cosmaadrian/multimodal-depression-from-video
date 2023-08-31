import os
import pandas as pd
from tqdm import tqdm

def compute_nframes(session_path, modality):
    session_modality_path = os.path.join(session_path, modality)
    chunks = sorted( os.listdir(session_modality_path) )
    nframes = int(chunks[-1].split(".")[0].split("_")[-1])

    return nframes

if __name__ == "__main__":
    audio_modality = "audio_mfcc"
    video_modality = "video_pose_gaze_aus"

    root_dir = "./data/"
    df_path = "./splits/training.csv"
    df = pd.read_csv(df_path)

    durations = []
    audio_frame_rates = []
    video_frame_rates = []
    for sessionID in tqdm(df["Participant_ID"].tolist()):
        session_path = os.path.join(root_dir, str(sessionID))

        # compute number of frames for both type of modalities
        audio_nframes = compute_nframes(session_path, audio_modality)
        video_nframes = compute_nframes(session_path, video_modality)

        # compute duration in seconds from the original data
        original_path = f"/home/dgimeno/phd/old-perceiving-depression/data/databases/E-DAIC-WOZ/corpus/data/{sessionID}/features/{sessionID}_OpenSMILE2.3.0_mfcc.csv"
        original_df = pd.read_csv(original_path, delimiter=";")
        duration = original_df["frameTime"].tolist()[-1]

        audio_frame_rate = audio_nframes / duration
        video_frame_rate = video_nframes / duration

        durations.append( duration )
        audio_frame_rates.append( audio_frame_rate )
        video_frame_rates.append( video_frame_rate )

    df["duration"] = durations
    df["audio_frame_rate"] = audio_frame_rates
    df["video_frame_rate"] = video_frame_rates
    df.to_csv(df_path.replace(".csv", "2.csv"), index = False)
