import models
MODELS = {
    'baseline': models.BaselineModel,
    'perceiver': models.PerceiverModel,
}

import datasets
DATASETS = {
    'd-vlog': datasets.DVlogDataset,
    'd-vlog-eval': datasets.DVlogEvaluationDataset,

    'original-d-vlog': datasets.OriginalDVlogDataset,
    'original-d-vlog-eval': datasets.OriginalDVlogEvaluationDataset,

    'original-d-vlog-new-split': datasets.OriginalDVlogNewSplitDataset,
    'original-d-vlog-new-split-eval': datasets.OriginalDVlogNewSplitEvaluationDataset,

    'daic-woz': datasets.DaicWozDataset,
    'daic-woz-eval': datasets.DaicWozEvaluationDataset,

    'e-daic-woz': datasets.EDaicWozDataset,
    'e-daic-woz-eval': datasets.EDaicWozEvaluationDataset,

}

MODALITY_ENCODERS = {
    'hand_landmarks': models.HandLandmarkEncoder,
    'face_landmarks': models.LandmarkEncoder,
    'body_landmarks': models.LandmarkEncoder,
    'audio_embeddings': models.NoOpEncoder,
    'face_embeddings': models.NoOpEncoder,
    'blinking_features': models.BlinkingEncoder,
    'gaze_features': models.NoOpEncoder,

    'orig_face_landmarks': models.LandmarkEncoder,
    'orig_audio_descriptors': models.NoOpEncoder,

    'daic_audio_covarep': models.NoOpEncoder,
    'daic_audio_formant': models.NoOpEncoder,
    'daic_facial_3d_landmarks': models.LandmarkEncoder,
    'daic_facial_aus': models.NoOpEncoder,
    'daic_facial_hog': models.NoOpEncoder,
    'daic_gaze': models.NoOpEncoder,
    'daic_head_pose': models.NoOpEncoder,

    'edaic_audio_egemaps': models.NoOpEncoder,
    'edaic_audio_mfcc': models.NoOpEncoder,
    'edaic_video_cnn_resnet': models.NoOpEncoder,
    'edaic_video_pose_gaze_aus': models.NoOpEncoder,

}

MODALITIES = {
    'face_landmarks': datasets.modalities.FaceLandmarks,
    'hand_landmarks': datasets.modalities.HandLandmarks,
    'body_landmarks': datasets.modalities.BodyLandmarks,
    'audio_embeddings': datasets.modalities.AudioEmbeddings,
    'face_embeddings': datasets.modalities.FaceEmbeddings,
    'blinking_features': datasets.modalities.BlinkingFeatures,
    'gaze_features': datasets.modalities.GazeFeatures,

    'orig_face_landmarks': datasets.modalities.OriginalDVlogFaceLandmarks,
    'orig_audio_descriptors': datasets.modalities.OriginalDVlogAudioDescriptors,

    'daic_audio_covarep': datasets.modalities.DaicWozAudioCovarep,
    'daic_audio_formant': datasets.modalities.DaicWozAudioFormant,
    'daic_facial_3d_landmarks': datasets.modalities.DaicWozFacial3dLandmarks,
    'daic_facial_aus': datasets.modalities.DaicWozFacialAus,
    'daic_facial_hog': datasets.modalities.DaicWozFacialHog,
    'daic_gaze': datasets.modalities.DaicWozGaze,
    'daic_head_pose': datasets.modalities.DaicWozHeadPose,

    'edaic_audio_egemaps': datasets.modalities.EDaicWozAudioEgemaps,
    'edaic_audio_mfcc': datasets.modalities.EDaicWozAudioMfcc,
    'edaic_video_cnn_resnet': datasets.modalities.EDaicWozVideoResnet,
    'edaic_video_pose_gaze_aus': datasets.modalities.EDaicWozPoseGazeAus,

}

import trainers
TRAINERS = {
    'classification': trainers.ClassificationTrainer,
}

import evaluators
EVALUATORS = {
    'temporal_evaluator': evaluators.TemporalEvaluator,
}

