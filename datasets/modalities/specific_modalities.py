from .base_modality import Modality
import numpy as np
from scipy import stats
import glob

class FaceLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'face_landmarks'

    def post_process(self, data):

        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        print('original shape')
        print(data.shape)

        data = np.asarray(np.split(data, self.args.n_temporal_windows, axis=0))

        padding_max = int(self.args.seconds_per_window * self.args.max_video_fps)

        if data.shape[0] < padding_max:
            padding = np.zeros((padding_max - data.shape[0], data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
            data = np.concatenate((data, padding), axis=0)
            
        print('output_shape')
        print(data.shape)

        return data


class HandLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'hand_landmarks'

    def post_process(self, data):

        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        return data


class BodyLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'body_pose_landmarks'

    def post_process(self, data):

        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        return data


class AudioEmbeddings(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'audio_pase_embeddings'

    def post_process(self, data):
        # TODO add padding and stuff like that
        return data


class FaceEmbeddings(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'face_emonet_embeddings'

    def post_process(self, data):

        return data