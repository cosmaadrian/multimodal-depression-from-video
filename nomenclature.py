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
    'hand_landmarks': models.modality_encoders.HandLandmarkEncoder,
    'face_landmarks': models.modality_encoders.LandmarkEncoder,
    'body_landmarks': models.modality_encoders.LandmarkEncoder,
    'audio_embeddings': models.modality_encoders.NoOpEncoder,
    'face_embeddings': models.modality_encoders.NoOpEncoder,
}

MODALITIES = {
    'face_landmarks': datasets.modalities.FaceLandmarks,
    'hand_landmarks': datasets.modalities.HandLandmarks,
    'body_landmarks': datasets.modalities.BodyLandmarks,
    'audio_embeddings': datasets.modalities.AudioEmbeddings,
    'face_embeddings': datasets.modalities.FaceEmbeddings,
}

import trainers
TRAINERS = {}

import evaluators
EVALUATORS = {
    'majority_classification': evaluators.MajorityClassificationEvaluator,
    # TODO add temporal evaluator
}

