import copy
import torch
import torch.nn as nn
from lib.trainer_extra.acumen_trainer import AcumenTrainer

class TemporalTrainer(AcumenTrainer):

    def __init__(self, args, model):
        super().__init__(args, model)

        from lib import nomenclature

        self.losses = {
            loss_args.target_head: nomenclature.LOSSES[loss_args.kind](self.args, loss_args = loss_args.args)
            for loss_args in self.args.losses
        }

    def training_step(self, batch, batch_idx):
        latent = None

        for window_idx in range(0, self.args.n_temporal_windows):
            window_batch = {}
            window_batch['video_frame_rate'] = batch['video_frame_rate']
            window_batch['audio_frame_rate'] = batch['audio_frame_rate']

            # look through temporal windows
            for modality in self.args.modalities:
                modality_id = modality.name
                window_batch[f"modality:{modality_id}:data"] = batch[f"modality:{modality_id}:data"][:, window_idx, ...]
                window_batch[f"modality:{modality_id}:mask"] = batch[f"modality:{modality_id}:mask"][:, window_idx, ...]

            outputs = self.model(window_batch, latent = latent)
            latent = outputs['latent']

        losses = 0
        for head_name, model_output in outputs.items():
            if head_name not in self.losses:
                continue

            loss = self.losses[head_name](y_true = batch['labels'], y_pred = model_output)
            losses = losses + loss

            self.log(f'train/loss:{head_name}', loss.item())

        final_loss = losses

        self.log('train/loss:final', final_loss.item(), on_step = True)
        return final_loss
