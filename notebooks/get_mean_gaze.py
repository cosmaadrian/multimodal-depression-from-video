import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_mean_gaze(video_gaze_dir):
    chunk_files = sorted( os.listdir(video_gaze_dir) )
    nframes = int(chunk_files[-1].split('.')[0].split('_')[-1])

    mean_gaze = np.zeros(3,)
    for chunk_file in chunk_files:
        chunk_path = os.path.join(video_gaze_dir, chunk_file)

        chunk = np.load(chunk_path)['data']
        mean_gaze = mean_gaze + np.mean(chunk, axis = 0)

    return mean_gaze

if __name__ == "__main__":
    gaze_dir = 'gaze_features'
    root_dir = '../data/databases/D-vlog/data/'

    for split_name in ["training", "validation", "test"]:
        split_path = f'../data/databases/D-vlog/splits/{split_name}.csv'

        # reading video IDs and their corresponding labels
        split_df = pd.read_csv(split_path, index_col=0)
        video_ids = split_df['video_id'].tolist()
        labels = split_df['label'].apply(lambda x: 1 if x == 'depression' else 0).tolist()

        # computing the average gaze features for each video ID
        mean_gazes = []
        for video_id in tqdm(video_ids):
            video_gaze_dir = os.path.join(root_dir, video_id, gaze_dir)
            mean_gazes.append( compute_mean_gaze(video_gaze_dir) )

        mean_gazes = np.array(mean_gazes)
        np.savez_compressed(f'./{split_name}_mean_gazes.npz', data=mean_gazes)
