import os
from .base_modality import Modality
import numpy as np
from scipy import stats
import glob

class FaceLandmarks(Modality):
    def __init__(self, df, args):
        self.df = df
        self.modality_dir = 'face_landmarks'
        self.modality_mask_file = 'no_face_idxs.npz'
        super().__init__(args)

class HandLandmarks(Modality):
    def __init__(self, df, args):
        self.df = df
        self.modality_dir = 'hand_landmarks'
        self.modality_mask_file = 'no_hand_idxs.npz'
        super().__init__(args)

class BodyLandmarks(Modality):
    def __init__(self, df, args):
        self.df = df
        self.modality_dir = 'body_pose_landmarks'
        self.modality_mask_file = 'no_body_idxs.npz'
        super().__init__(args)

class AudioEmbeddings(Modality):
    def __init__(self, df, args):
        self.df = df
        self.modality_dir = 'audio_pase_embeddings'
        self.modality_mask_file = 'no_voice_idxs.npz'
        super().__init__(args)

class FaceEmbeddings(Modality):
    def __init__(self, df, args):
        self.df = df
        self.modality_dir = 'face_emonet_embeddings'
        self.modality_mask_file = 'no_face_idxs.npz'
        super().__init__(args)
