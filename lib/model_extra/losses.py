from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss

import torch
import math
from torch import nn

from lib import device

class CoralLoss(torch.nn.Module):
	def __init__(self, args, loss_args):
		super(CoralLoss, self).__init__()
		self.args = args
		self.loss_args = loss_args

	def forward(self, y_true, y_pred, **kwargs):
		levels = levels_from_labelbatch(y_true.view(-1).long(), num_classes = self.loss_args.num_classes).to(device)
		return coral_loss(y_pred.logits, levels)

class AcumenCrossEntropy(torch.nn.Module):
    def __init__(self, args, loss_args):
        super(AcumenCrossEntropy, self).__init__()
        self.args = args
        self.loss_args = loss_args

        if 'weights' in self.loss_args:
            self.weights = torch.tensor(self.loss_args.weights).float().to(device)
        else:
            self.weights = None

    def forward(self, y_true, y_pred, **kwargs):
        return nn.functional.cross_entropy(
            input = y_pred.logits,
            target = y_true.view(-1).long(),
            weight = self.weights,
        )

class AcumenBinaryCrossEntropy(torch.nn.Module):
    def __init__(self, args, loss_args):
        super(AcumenBinaryCrossEntropy, self).__init__()
        self.args = args
        self.loss_args = loss_args

        if 'weights' in self.loss_args:
            self.weights = torch.tensor(self.loss_args.weights).float().to(device)
        else:
            self.weights = None

    def forward(self, y_true, y_pred, **kwargs):
        return nn.functional.binary_cross_entropy(
            input = y_pred.logits.view(-1),
            target = y_true.view(-1),
            weight = self.weights,
        )
