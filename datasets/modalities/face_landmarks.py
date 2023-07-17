from .base_modality import Modality
import numpy as np

class FaceLandmarks(Modality):
    def __init__(self, args):
        super().__init__(args)

    def read_chunk(self, window):
        # TODO needs refactoring
        data = np.load(f'{self.args.environment["d-vlog"]}/data/face_landmarks/{window}')['data']
        data = self.post_process(data)

        data = data[:256].astype(np.float32)

        return data

