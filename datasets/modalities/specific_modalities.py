from .base_modality import Modality
import numpy as np
from scipy import stats
import glob

class FaceLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'face_landmarks'


class HandLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'hand_landmarks'


class BodyLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'body_pose_landmarks'


class AudioEmbeddings(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'audio_pase_embeddings'


class FaceEmbeddings(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'face_emonet_embeddings'
