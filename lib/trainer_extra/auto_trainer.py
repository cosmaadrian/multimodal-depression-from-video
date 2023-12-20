import torch
import torch.nn as nn
from .acumen_trainer import AcumenTrainer

class AutoTrainer(AcumenTrainer):

    def __init__(self, args, model):
        super().__init__(args, model)

        from lib import nomenclature

        self.losses = {
            loss_args.target_head: nomenclature.LOSSES[loss_args.kind](self.args, loss_args = loss_args.args)
            for loss_args in self.args.losses
        }

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        losses = 0

        for head_name, model_output in outputs.items():
            if head_name not in self.losses:
                continue
            loss = self.losses[head_name](y_true = batch['labels'], y_pred = model_output)
            losses = losses + loss

            self.log(f'train/loss:{head_name}', loss.item())

        final_loss = losses / len(outputs.keys())

        self.log('train/loss:final', final_loss.item(), on_step = True)
        return final_loss
