import models
MODELS = {
    'baseline': models.BaselineModel,
    'perceiver': models.PerceiverModel
}

import datasets
DATASETS = {
    'd-vlog': datasets.DVlogDataset
}

MODALITY_ENCODERS = {
    'hand_landmarks': models.HandLandmarkEncoder,
    'face_landmarks': models.LandmarkEncoder,
    'body_landmarks': models.LandmarkEncoder,
    'audio_embeddings': models.NoOpEncoder,
    'face_embeddings': models.NoOpEncoder,
    'blinking_patterns': models.BlinkingEncoder,
    'gaze_patterns': models.NoOpEncoder,
}

MODALITIES = {
    'face_landmarks': datasets.modalities.FaceLandmarks,
    'hand_landmarks': datasets.modalities.HandLandmarks,
    'body_landmarks': datasets.modalities.BodyLandmarks,
    'audio_embeddings': datasets.modalities.AudioEmbeddings,
    'face_embeddings': datasets.modalities.FaceEmbeddings,
    'blinking_patterns': datasets.modalities.BlinkingPatterns,
    'gaze_patterns': datasets.modalities.GazePatterns,
}

import trainers
TRAINERS = {
    'temporal': trainers.TemporalTrainer
}

import evaluators
EVALUATORS = {
    'majority_classification': evaluators.MajorityClassificationEvaluator,
    # TODO add temporal evaluator
}

