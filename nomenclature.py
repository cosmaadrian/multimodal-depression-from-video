import models
MODELS = {
    'baseline': models.BaselineModel,
    'perceiver': models.PerceiverModel,
}

import datasets
DATASETS = {
    'd-vlog': datasets.DVlogDataset,
    'd-vlog-eval': datasets.DVlogEvaluationDataset,
    
    # Original DVlog data
    'original-d-vlog': datasets.OriginalDVlogDataset,
    'original-d-vlog-eval': datasets.OriginalDVlogEvaluationDataset,
}

MODALITY_ENCODERS = {
    'hand_landmarks': models.HandLandmarkEncoder,
    'face_landmarks': models.LandmarkEncoder,
    'body_landmarks': models.LandmarkEncoder,
    'audio_embeddings': models.NoOpEncoder,
    'face_embeddings': models.NoOpEncoder,
    'blinking_features': models.BlinkingEncoder,
    'gaze_features': models.NoOpEncoder,

    # Original DVlog modalities
    'orig_face_landmarks': models.LandmarkEncoder,
    'orig_audio_descriptors': models.NoOpEncoder,
}

MODALITIES = {
    'face_landmarks': datasets.modalities.FaceLandmarks,
    'hand_landmarks': datasets.modalities.HandLandmarks,
    'body_landmarks': datasets.modalities.BodyLandmarks,
    'audio_embeddings': datasets.modalities.AudioEmbeddings,
    'face_embeddings': datasets.modalities.FaceEmbeddings,
    'blinking_features': datasets.modalities.BlinkingFeatures,
    'gaze_features': datasets.modalities.GazeFeatures,

    # Original DVlog modalities
    'orig_face_landmarks': datasets.modalities.OriginalDVlogFaceLandmarks,
    'orig_audio_descriptors': datasets.modalities.OriginalDVlogAudioDescriptors,
}

import trainers
TRAINERS = {
    'temporal': trainers.TemporalTrainer,
    'classification': trainers.ClassificationTrainer,
}

import evaluators
EVALUATORS = {
    'majority_evaluator': evaluators.MajorityClassificationEvaluator,
    'temporal_evaluator': evaluators.TemporalEvaluator,
}

