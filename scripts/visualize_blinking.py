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

data_path = '/home/dgimeno/phd/perceiving-depression/data/databases/D-vlog/data/'
files = glob.glob(data_path + '*')

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

depressed_blinking = []
non_depressed_blinking = []

blinking_rate_depressed = []
blinking_rate_non_depressed = []

def get_data(df):
    count = 0
    for i, video_id in enumerate(tqdm.tqdm(files)):
        current_meta = df[df['video_id'] == video_id.split('/')[-1]]
        # is depressed?
        is_depressed = current_meta['label'].values

        if len(is_depressed) == 0:
            continue

        no_blink = np.load(video_id + '/no_blink_idxs.npz')['data']
        inverse_no_blink = np.arange(current_meta['video_nframes'].values[0])
        inverse_no_blink = np.delete(inverse_no_blink, no_blink)

        chunks = glob.glob(video_id + '/blinking_features/*.npz')
        chunks = sorted(chunks, key = lambda x: int(x.split('_')[-2]))

        current_blinking = []

        for chunk in chunks:
            x = np.load(chunk)['data']
            current_blinking.extend(x)

        current_blinking = np.array(current_blinking)

        blinking_indices = np.argwhere(current_blinking == 1).ravel()
        blinking_rate = np.diff(blinking_indices) / current_meta['video_frame_rate'].values[0]

        average_blinking_rate = np.mean(blinking_rate)
        std_blinking_rate = np.std(blinking_rate)

        if is_depressed[0] == 'depression':
            blinking_rate_depressed.append([average_blinking_rate, std_blinking_rate])
        else:
            blinking_rate_non_depressed.append([average_blinking_rate, std_blinking_rate])


    return blinking_rate_depressed, blinking_rate_non_depressed

# blinking_rate_depressed, blinking_rate_non_depressed = get_data(df)

# blinking_rate_depressed = np.array(blinking_rate_depressed)
# blinking_rate_non_depressed = np.array(blinking_rate_non_depressed)

# np.savez_compressed('cache/blinking.npz', depressed = blinking_rate_depressed, nondepressed = blinking_rate_non_depressed)

from scipy.stats import poisson

data = np.load('cache/blinking.npz')
depressed_blinking = data['depressed']
non_depressed_blinking = data['nondepressed']

# remove nans
depressed_blinking = depressed_blinking[~np.isnan(depressed_blinking).any(axis=1)]
non_depressed_blinking = non_depressed_blinking[~np.isnan(non_depressed_blinking).any(axis=1)]

k = np.arange(0, 5, 1)
lambda_depressed = (np.mean(depressed_blinking[:, 0]) + np.mean(depressed_blinking[:, 1])) / 2
lambda_non_depressed = (np.mean(non_depressed_blinking[:, 0]) + np.mean(non_depressed_blinking[:, 1])) / 2

print(lambda_depressed, lambda_non_depressed)
p_depressed = poisson.pmf(k, lambda_depressed)
p_non_depressed = poisson.pmf(k, lambda_non_depressed)

plt.plot(k, p_depressed, 'bo-', label='depressed')
plt.plot(k, p_non_depressed, 'go-', label='non depressed')
plt.legend()
plt.savefig('blinking_poisson.png')

# print(depressed_blinking.shape)
# print(non_depressed_blinking.shape)

# plt.scatter(depressed_blinking[:, 0], depressed_blinking[:, 1], label = 'depressed')
# plt.scatter(non_depressed_blinking[:, 0], non_depressed_blinking[:, 1], label = 'non depressed')
# plt.legend()
# plt.savefig('blinking.png')