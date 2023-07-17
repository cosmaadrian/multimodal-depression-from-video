from .base_modality import Modality
import numpy as np

class FaceLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)

    def read_chunk(self, window):
        # TODO needs refactoring
        data = np.load(f'../../data/databases/D-vlog/data/face_landmarks/{window}')
        data = self.post_process(data)

        return data

