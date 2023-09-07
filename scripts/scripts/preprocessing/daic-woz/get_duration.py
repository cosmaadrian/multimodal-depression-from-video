import os
import sys
import pandas as pd

def compute_nframes(session_path):
    chunks = sorted( os.listdir(session_path) )
    nframes = int(chunks[-1].split(".")[0].split("_")[-1])

    return nframes

def update_split(split_path, durations):
   split = pd.read_csv(split_path)
   split = split.drop(columns = ["Unnamed: 0"])
   split["duration"] = durations

   split.to_csv(split_path.replace(".csv", "2.csv"), index = False)


if __name__ == "__main__":
    split_set = sys.argv[1]

    data_dir = "./data/"
    modality = "audio_covarep"
    sessionIDs = sorted( os.listdir(data_dir) )

    split_path = f"./splits/{split_set}.csv"
    splitIDs = pd.read_csv(split_path)["Participant_ID"].apply(lambda x: str(x)).tolist()

    durations = []
    total_seconds = 0.0
    for sessionID in sessionIDs:
        session_path = os.path.join(data_dir, sessionID, modality)
        session_nframes = compute_nframes(session_path)
        session_nseconds = session_nframes / 100.0

        if sessionID in splitIDs:
            total_seconds += session_nseconds
            durations.append(session_nseconds)

    update_split(split_path, durations)
    total_hours = (total_seconds / 60.0) / 60.0
    print(f"Total hours: {total_hours} || Number of samples: {len(splitIDs)} || Total seconds: {total_seconds}")
