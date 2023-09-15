import json
from easydict import EasyDict

from lib import nomenclature, device
import torch
from lib import device

from lib.arg_utils import define_args
from lib.utils import load_model, load_config
from lib.loggers import NoLogger
from lib.forge import VersionCommand

import tqdm

from captum.attr import IntegratedGradients

class ModelAdaptor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self,
        modality_hand_landmarks_data,

        modality_face_landmarks_data,

        modality_body_landmarks_data,

        modality_audio_embeddings_data,

        modality_face_embeddings_data,

        modality_blinking_features_data,

        modality_gaze_features_data,
    ):
        model_input = {
            'modality:hand_landmarks:data': modality_hand_landmarks_data,
            # 'modality:hand_landmarks:mask': modality_hand_landmarks_mask,

            'modality:face_landmarks:data': modality_face_landmarks_data,
            # 'modality:face_landmarks:mask': modality_face_landmarks_mask,

            'modality:body_landmarks:data': modality_body_landmarks_data,
            # 'modality:body_landmarks:mask': modality_body_landmarks_mask,

            'modality:audio_embeddings:data': modality_audio_embeddings_data,
            # 'modality:audio_embeddings:mask': modality_audio_embeddings_mask,

            'modality:face_embeddings:data': modality_face_embeddings_data,
            # 'modality:face_embeddings:mask': modality_face_embeddings_mask,

            'modality:blinking_features:data': modality_blinking_features_data,
            # 'modality:blinking_features:mask': modality_blinking_features_mask,

            'modality:gaze_features:data': modality_gaze_features_data,
            # 'modality:gaze_features:mask': modality_gaze_features_mask,
        }

        masks = {}
        for key, value in model_input.items():
            if 'data' not in key:
                continue
            seq_len = value.shape[1]
            the_mask = torch.ones((value.shape[0], value.shape[1])).bool()
            masks[key.replace('data', 'mask')] = the_mask

        model_input.update(masks)

        model_input['audio_frame_rate'] = 100 * torch.ones(model_input['modality:gaze_features:data'].shape[0]).to(device)
        model_input['video_frame_rate'] = 30 * torch.ones(model_input['modality:gaze_features:data'].shape[0]).to(device)

        outputs = self.model(model_input)
        return outputs['depression'].probas[:, 1]

VersionCommand().run()

args = define_args(
    require_config_file = False,
    extra_args = [
        ('--checkpoint_kind', {'default': 'best', 'type': str, 'required': False}),
        ('--batch_size', {'default': 1, 'type': int, 'required': False})
    ])
args.batch_size = 1

# checkpoints/dvlog-baseline-ablation-modalities-final:av-lm-eyes-spw-9-pt-0.50-run-1
args.group = 'dvlog-baseline-ablation-modalities-final'
args.name = 'av-lm-eyes-spw-9-pt-0.50-run-1'

# checkpoints/dvlog-baseline-ablation-modalities-final:av-eyes-spw-9-pt-0.50-run-4
# args.group = 'dvlog-baseline-ablation-modalities-final'
# args.name = 'av-eyes-spw-9-pt-0.50-run-4'

model_config = load_config(args)

args = EasyDict({**model_config, **args})

args.modalities = [modality for modality in args.modalities if modality.name in args.use_modalities]

architecture = nomenclature.MODELS[args.model](args)

state_dict = load_model(args, map_location = device)
state_dict = {
    key.replace('module.', ''): value
    for key, value in state_dict.items()
}

architecture.load_state_dict(state_dict)
architecture.to(device)

model_adaptor = ModelAdaptor(model = architecture)
ig = IntegratedGradients(model_adaptor)

###################################
dataset = nomenclature.DATASETS['d-vlog']
val_dataloader = dataset.val_dataloader(args, kind = 'test')
###################################

# keys_of_interest = sum([
#     [f'modality:{key}:data', f'modality:{key}:mask']
#     for key in args.use_modalities
# ], [])
# keys_of_interest.extend(['video_frame_rate', 'audio_frame_rate'])

keys_of_interest = [
    'modality:hand_landmarks:data',
    'modality:face_landmarks:data',
    'modality:body_landmarks:data',
    'modality:audio_embeddings:data',
    'modality:face_embeddings:data',
    'modality:blinking_features:data',
    'modality:gaze_features:data',
]

all_attributions = {}
for i, batch in enumerate(tqdm.tqdm(val_dataloader, total = len(val_dataloader))):
    latent = None
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

        if 'modality' in key:
            batch[key] = batch[key].squeeze(1)

    inputs = tuple([
        batch[key]
        for key in keys_of_interest
     ])

    baselines = tuple([
        torch.zeros_like(input_)
        for input_ in inputs
    ])

    # print(":: Extracting attributions")
    attributions = ig.attribute(
        inputs,
        baselines,
        n_steps = 16,
        return_convergence_delta = False,
    )

    attributions_dict = {}
    for i, key in enumerate(keys_of_interest):
        if attributions[i] is None:
            continue

        axes = tuple(range(2, len(attributions[i].shape)))

        current_attribution = attributions[i].sum(axis = axes).squeeze(0)
        current_attribution = current_attribution / torch.norm(current_attribution)
        current_attribution = current_attribution.detach().cpu().numpy().tolist()

        attributions_dict[key] = current_attribution

    all_attributions[batch['video_id'][0]] = attributions_dict

with open('results/explainability/attributions.json', 'wt') as file:
    json.dump(all_attributions, file)
