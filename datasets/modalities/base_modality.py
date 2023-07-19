import glob
import numpy as np

class Modality(object):
    def __init__(self, args):
        self.args = args
        self.chunk_cache = {}

    def _indexes_from_chunkfiles(self, video_id):
        if video_id in self.chunk_cache:
            return self.chunk_cache[video_id]

        chunk_files = glob.glob(f'{self.args.environment["d-vlog"]}/data/{video_id}/{self.modality_dir}/*.npz')
        indexes = [(int(chunk_file.split('/')[-1].split('.')[0].split('_')[-2]), int(chunk_file.split('/')[-1].split('.')[0].split('_')[-1])) for chunk_file in chunk_files]
        self.chunk_cache[video_id] = indexes

        return indexes

    def read_chunk(self, video, start, end):
        video_id = video["video_id"]
        fps =  100 if "audio" in self.modality_dir else video["video_frame_rate"]

        start_frame = int(start * fps)
        end_frame = int(end * fps) 

        indexes = self._indexes_from_chunkfiles(video_id)
        
        min_index = min([v for v in indexes if v[0] <= start_frame], key = lambda x: abs(x[0] - start_frame))[0]
        max_index = min([v for v in indexes if v[1] >= end_frame], key = lambda x: abs(x[1] - end_frame))[1]

        files_in_window = sorted([v for v in indexes if v[0] >= min_index and v[1] <= max_index])

        for i, (start, end) in enumerate(files_in_window):
            data = np.load(f'{self.args.environment["d-vlog"]}/data/{video_id}/{self.modality_dir}/{video_id}_{str(start).zfill(6)}_{str(end).zfill(6)}.npz')['data']

            start_index = 0
            end_index = data.shape[0]

            if start_frame > start:
                start_index = start_frame - start
            
            if end_frame < end:
                end_index = end_frame - start

            if i == 0:
                output = data[start_index:end_index]
            else:
                output = np.concatenate((output, data[start_index:end_index]))

            output = self.post_process(output)

        return output

    def post_process(self, data):
        # TODO add padding and stuff like that
        return data
