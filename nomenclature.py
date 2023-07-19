import models
MODELS = {
    'perceiver': models.PerceiverModel
}

import datasets
DATASETS = {
    'd-vlog': datasets.DVlogDataset
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
    # TODO
    'classification': evaluators.ClassificationEvaluator,
}

