import pandas as pd


def compute_class_weights():
    dir = '/home/banamar/perceiving-depression/data/databases/DAIC-WOZ/'
    df = pd.read_csv(f'{dir}/splits/training.csv')
    print(df)

    frames_depression = df[df['PHQ8_Binary'] == 1]['duration'].sum()
    frames_no_depression = df[df['PHQ8_Binary'] == 0]['duration'].sum()

    print(f'Duration depression: {frames_depression}')
    print(f'Duration no depression: {frames_no_depression}')

    n_samples = frames_depression + frames_no_depression
    n_classes = 2

    weights = [n_samples / (n_classes * frames_no_depression), n_samples / (n_classes * frames_depression)]

    print(f'Class weights: {weights}')


if __name__ == '__main__':
    compute_class_weights()