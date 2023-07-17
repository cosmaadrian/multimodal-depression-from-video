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
    # TODO add more
}

import trainers
TRAINERS = {}

import evaluators
EVALUATORS = {
    # TODO
    'classification': evaluators.ClassificationEvaluator,
}

