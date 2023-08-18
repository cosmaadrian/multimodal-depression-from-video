import matplotlib.pyplot as plt
import ndtest
import seaborn as sns
import numpy as np
import pandas as pd
import tqdm
import seaborn as sns
import glob

df = pd.concat([
    pd.read_csv('../data/databases/D-vlog/splits/training.csv')
])

def gaze_to_angles(gaze_vector):
    pitch = np.arcsin(-gaze_vector[:, 1])
    yaw = np.arctan2(-gaze_vector[:, 0], -gaze_vector[:, 2])
    return pitch, yaw

# Read in the data
data_path = '/home/dgimeno/phd/perceiving-depression/data/databases/D-vlog/data/'

# Get the list of files
files = glob.glob(data_path + '*')

pitches = []
yaws = []


def get_data(depression_label):
    count = 0
    for i, video_id in enumerate(tqdm.tqdm(files)):
        # is depressed?
        is_depressed = df[df['video_id'] == video_id.split('/')[-1]]['label'].values
        if len(is_depressed) == 0:
            continue

        if is_depressed[0] != depression_label:
            continue

        chunks = glob.glob(video_id + '/gaze_features/*.npz')

        current_pitches = []
        current_yaws = []

        for chunk in chunks:
            x = np.load(chunk)['data']

            # remove zeros
            x = x[~np.all(x == 0, axis=1)]

            pitch, yaw = gaze_to_angles(x)

            current_pitches.extend(pitch)
            current_yaws.extend(yaw)

        # sample 1000 random values
        current_pitches = np.random.choice(current_pitches, 100)
        current_yaws = np.random.choice(current_yaws, 100)

        pitches.extend(current_pitches)
        yaws.extend(current_yaws)

        count += 1

    return pitches, yaws


def plot_stuff(depression_label):
    pitches, yaws = get_data(depression_label)

    return pitches, yaws
    # fig, ax = plt.subplots()
    # pitches_radians = np.array(pitches)
    # yaws_radians = np.array(yaws)
    # sns.kdeplot(x = pitches_radians, y = yaws_radians, cmap='coolwarm', fill=True, ax=ax, thresh=0, levels=100)

    # ax.set_xlim(-np.pi / 2, np.pi / 2)
    # ax.set_ylim(-np.pi / 2, np.pi / 2)

    # ax.grid(True)
    # ax.set_title(f'Heatmap for {depression_label.capitalize()} videos')
    # plt.savefig(f'polar_heatmap_{depression_label}.png')

x1, y1 = plot_stuff('depression')
x2, y2 = plot_stuff('normal')

P, D = ndtest.ks2d2s(x1, y1, x2, y2, extra=True)
print(f"{P=:.3g}, {D=:.3g}")