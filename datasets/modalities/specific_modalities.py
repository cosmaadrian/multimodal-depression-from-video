from .base_modality import Modality
import numpy as np
from scipy import stats
import glob

class FaceLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'face_landmarks'
        self.modality_mask_file = 'no_face_idxs.npz'


class HandLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'hand_landmarks'
        self.modality_mask_file = 'no_hand_idxs.npz'


class BodyLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'body_pose_landmarks'
        self.modality_mask_file = 'no_body_idxs.npz'



class AudioEmbeddings(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'audio_pase_embeddings'
        self.modality_dir = 'voice_activity.npz'


class FaceEmbeddings(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.modality_dir = 'face_emonet_embeddings'
        self.modality_mask_file = 'no_face_idxs.npz'
