from .base_modality import Modality
import numpy as np
import glob

class FaceLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.fps = 25
        self.modality_dir = 'face_landmarks'

    def post_process(self, data):
        # TODO reshape / normalize etc
        return data


class HandLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.fps = 25
        self.modality_dir = 'hand_landmarks'

    def post_process(self, data):
        # TODO reshape / normalize etc
        return data

class BodyLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.fps = 25
        self.modality_dir = 'body_pose_landmarks'

    def post_process(self, data):
        # TODO reshape / normalize etc
        return data

class AudioEmbeddings(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.fps = 100
        self.modality_dir = 'audio_pase_embeddings'

    def post_process(self, data):
        # TODO reshape / normalize etc
        return data

class FaceEmbeddings(Modality):
    def __init__(self, args):
        super().__init__(args)
        self.fps = 25
        self.modality_dir = 'face_embeddings'

    def post_process(self, data):
        # TODO reshape / normalize etc
        return data