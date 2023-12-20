import easydict

from .model_extra import CoralHead, ClassificationHead, CoralLoss, AcumenBinaryCrossEntropy, AcumenCrossEntropy, MultiLabelHead
from .trainer_extra import *

NOMENCLATURE = easydict.EasyDict({
	'TRAINERS': {
	},

	'HEADS': {
		'classification': ClassificationHead,
	},

	'LOSSES': {
        'xe': AcumenCrossEntropy,
	},

	'DATASETS': {
	},

	'MODELS': {
	},

	'EVALUATORS': {

	},

	'MODALITIES': {
	},
	'MODALITY_ENCODERS': {

	},
})



import nomenclature

for actor_type in ['MODELS', 'TRAINERS', 'DATASETS', 'EVALUATORS', 'HEADS', 'MODALITIES', 'LOSSES', 'MODALITY_ENCODERS']:
	if actor_type not in nomenclature.__dict__:
		continue

	for key, value in nomenclature.__dict__[actor_type].items():
		if key in NOMENCLATURE[actor_type]:
			raise Exception(f'::: {key} already defined for {actor_type}.')

		NOMENCLATURE[actor_type][key] = value

