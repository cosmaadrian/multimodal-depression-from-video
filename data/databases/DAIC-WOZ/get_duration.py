import os
import pandas as pd

def compute_nframes(session_path):
    chunks = sorted( os.listdir(session_path) )
    nframes = int(chunks[-1].split(".")[0].split("_")[-1])

    return nframes

if __name__ == "__main__":
    data_dir = "./data/"
    modality = "audio_covarep"
    sessionIDs = sorted( os.listdir(data_dir) )

    split_path = "./splits/test.csv"
    splitIDs = pd.read_csv(split_path)["Participant_ID"].apply(lambda x: str(x)).tolist()

    total_seconds = 0.0
    for sessionID in sessionIDs:
        session_path = os.path.join(data_dir, sessionID, modality)
        session_nframes = compute_nframes(session_path)
        session_nseconds = session_nframes / 100.0

        if sessionID in splitIDs:
            total_seconds += session_nseconds

    total_hours = (total_seconds / 60.0) / 60.0
    print(f"Total hours: {total_hours} || Number of samples: {len(splitIDs)}")
