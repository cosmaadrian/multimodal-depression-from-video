import os
from .base_modality import Modality
import numpy as np
from scipy import stats
import glob

class FaceLandmarks(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'face_landmarks'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "face_emonet_embeddings"
        super().__init__(args)

class HandLandmarks(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'hand_landmarks'
        self.modality_mask_file = 'no_hand_idxs.npz'
        self.video_ref_modality = "face_emonet_embeddings"
        super().__init__(args)

class BodyLandmarks(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'body_pose_landmarks'
        self.modality_mask_file = 'no_body_idxs.npz'
        self.video_ref_modality = "face_emonet_embeddings"
        super().__init__(args)

class AudioEmbeddings(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'audio_pase_embeddings'
        self.modality_mask_file = 'no_voice_idxs.npz'
        self.video_ref_modality = "face_emonet_embeddings"
        super().__init__(args)

class FaceEmbeddings(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'face_emonet_embeddings'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "face_emonet_embeddings"
        super().__init__(args)

class GazeFeatures(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'gaze_features'
        self.modality_mask_file = 'no_gaze_idxs.npz'
        self.video_ref_modality = "face_emonet_embeddings"
        super().__init__(args)

class BlinkingFeatures(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'blinking_features'
        self.modality_mask_file = 'no_blink_idxs.npz'
        self.video_ref_modality = "face_emonet_embeddings"
        super().__init__(args)

class OriginalDVlogFaceLandmarks(Modality):
    def __init__(self, df, env_path, args):
        self.args = args
        self.df = df
        self.env_path = env_path

    def post_process(self, data, no_modality_mask):
        return data, np.ones(data.shape[0]).astype(bool)

    def read_chunk(self, video_sample, start_in_seconds, end_in_seconds):
        video_features = np.load(f'{self.args.environment["d-vlog-original"]}/{video_sample["video_id"]}/{video_sample["video_id"]}_visual.npy')
        if end_in_seconds > video_features.shape[0]:
            end_in_seconds = video_features.shape[0]
            start_in_seconds = end_in_seconds - self.args.seconds_per_window

        output = video_features[start_in_seconds:end_in_seconds]
        output = np.expand_dims(output, self.args.n_temporal_windows)

        return output.astype('float32'), np.ones(output.shape[0])

class OriginalDVlogAudioDescriptors(Modality):
    def __init__(self, df, env_path, args):
        self.args = args
        self.df = df
        self.env_path = env_path

    def post_process(self, data, no_modality_mask):
        return data, np.ones(data.shape[0]).astype(bool)

    def read_chunk(self, video_sample, start_in_seconds, end_in_seconds):
        audio_descriptors = np.load(f'{self.args.environment["d-vlog-original"]}/{video_sample["video_id"]}/{video_sample["video_id"]}_acoustic.npy')
        if end_in_seconds > audio_descriptors.shape[0]:
            end_in_seconds = audio_descriptors.shape[0]
            start_in_seconds = end_in_seconds - self.args.seconds_per_window

        output = audio_descriptors[start_in_seconds:end_in_seconds]
        output = np.expand_dims(output, self.args.n_temporal_windows)

        return output.astype('float32'), np.ones(output.shape[0])

class DaicWozAudioCovarep(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'audio_covarep'
        self.modality_mask_file = 'no_voice_idxs.npz'
        self.video_ref_modality = "facial_3d_landmarks"
        super().__init__(args)

class DaicWozAudioFormant(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'audio_formant'
        self.modality_mask_file = 'no_voice_idxs.npz'
        self.video_ref_modality = "facial_3d_landmarks"
        super().__init__(args)

class DaicWozFacial3dLandmarks(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'facial_3d_landmarks'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "facial_3d_landmarks"
        super().__init__(args)

class DaicWozFacialAus(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'facial_aus'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "facial_3d_landmarks"
        super().__init__(args)

class DaicWozFacialHog(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'facial_hog'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "facial_3d_landmarks"
        super().__init__(args)

class DaicWozGaze(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'gaze'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "facial_3d_landmarks"
        super().__init__(args)

class DaicWozHeadPose(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'head_pose'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "facial_3d_landmarks"
        super().__init__(args)

class EDaicWozAudioEgemaps(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'audio_egemaps'
        self.modality_mask_file = 'no_voice_idxs.npz'
        self.video_ref_modality = "video_pose_gaze_aus"
        super().__init__(args)

class EDaicWozAudioMfcc(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'audio_mfcc'
        self.modality_mask_file = 'no_voice_idxs.npz'
        self.video_ref_modality = "video_pose_gaze_aus"
        super().__init__(args)

class EDaicWozVideoResnet(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'video_cnn_resnet'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "video_pose_gaze_aus"
        super().__init__(args)

class EDaicWozPoseGazeAus(Modality):
    def __init__(self, df, env_path, args):
        self.df = df
        self.env_path = env_path
        self.modality_dir = 'video_pose_gaze_aus'
        self.modality_mask_file = 'no_face_idxs.npz'
        self.video_ref_modality = "video_pose_gaze_aus"
        super().__init__(args)
