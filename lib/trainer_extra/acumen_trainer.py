import torch
import torch.nn as nn
import copy

class AcumenTrainer(object):
    def __init__(self, args, model):
        self.args = self.args = copy.deepcopy(args)
        self.model = model
        self._optimizer = None

    def configure_optimizers(self, lr = 0.1):
        if self._optimizer is not None:
            return self._optimizer

        self._optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.model.parameters()), lr)

        return self._optimizer

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def validation_epoch_start(self, outputs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, epoch = None):
        pass

    def training_batch_start(self, batch = None):
        pass

    def training_batch_end(self, batch = None):
        pass

    def training_epoch_start(self, epoch = None):
        pass

    def training_end(self):
        pass

    def training_start(self):
        pass
