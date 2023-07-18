import models
MODELS = {
    'perceiver': models.PerceiverModel
}

import datasets
DATASETS = {
    'd-vlog': datasets.DVlogDataset
}

MODALITIES = {
    'face-landmarks': datasets.modalities.FaceLandmarks,
    'hand-landmarks': datasets.modalities.HandLandmarks,
    'body-landmarks': datasets.modalities.BodyLandmarks,
    'audio-embeddings': datasets.modalities.AudioEmbeddings,
    'face-embeddings': datasets.modalities.FaceEmbeddings,
}

import trainers
TRAINERS = {}

import evaluators
EVALUATORS = {
    # TODO
    'classification': evaluators.ClassificationEvaluator,
}

