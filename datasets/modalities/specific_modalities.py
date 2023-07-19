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
        # TODO reshape / normalize etc
        return data


class HandLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'hand_landmarks'

    def post_process(self, data):
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        # TODO reshape / normalize etc
        return data


class BodyLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'body_pose_landmarks'

    def post_process(self, data):
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        # TODO reshape / normalize etc
        return data


class AudioEmbeddings(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'audio_pase_embeddings'

    def post_process(self, data):
        # TODO reshape / normalize etc
        return data


class FaceEmbeddings(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'face_emonet_embeddings'

    def post_process(self, data):
        # TODO reshape / normalize etc
        return data