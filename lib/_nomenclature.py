import easydict

from .model_extra import CoralHead, ClassificationHead, CoralLoss, AcumenBinaryCrossEntropy, AcumenCrossEntropy, MultiLabelHead
from .trainer_extra import *

NOMENCLATURE = easydict.EasyDict({
	'TRAINERS': {
		'auto': AutoTrainer,
	},

	'HEADS': {
		'coral': CoralHead,
		'classification': ClassificationHead,
		'multilabel': MultiLabelHead,
	},

	'LOSSES': {
		'coral': CoralLoss,
        'xe': AcumenCrossEntropy,
        'bce': AcumenBinaryCrossEntropy,
	},

	'DATASETS': {
	},

	'MODELS': {
	},

	'EVALUATORS': {
        # TODO classification evaluator
	},
})



import nomenclature

# Merging with user stuff.

for actor_type in ['MODELS', 'TRAINERS', 'DATASETS', 'EVALUATORS', 'HEADS']:
	if actor_type not in nomenclature.__dict__:
		continue

	for key, value in nomenclature.__dict__[actor_type].items():
		if key in NOMENCLATURE[actor_type]:
			raise Exception(f'::: {key} already defined for {actor_type}.')

		NOMENCLATURE[actor_type][key] = value

