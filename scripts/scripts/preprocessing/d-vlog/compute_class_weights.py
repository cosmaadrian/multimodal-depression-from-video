import pandas as pd


def compute_class_weights():
    dir = '/home/dgimeno/phd/perceiving-depression/data/databases/D-vlog/'
    df = pd.read_csv(f'{dir}/splits/training.csv', index_col=0)

    frames_depression = df[df['label'] == 'depression']['video_nframes'].sum()
    frames_no_depression = df[df['label'] == 'normal']['video_nframes'].sum()

    print(f'Frames depression: {frames_depression}')
    print(f'Frames no depression: {frames_no_depression}')

    n_samples = frames_depression + frames_no_depression
    n_classes = 2

    weights = [n_samples / (n_classes * frames_no_depression), n_samples / (n_classes * frames_depression)]

    print(f'Class weights: {weights}')



if __name__ == '__main__':
    compute_class_weights()