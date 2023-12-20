import torch
import torch.nn as nn

from coral_pytorch.dataset import proba_to_label
from coral_pytorch.layers import CoralLayer


from dataclasses import dataclass
from typing import NamedTuple

class ClassificationOutput(NamedTuple):
    logits: torch.Tensor
    probas: torch.Tensor
    labels: torch.Tensor

class ModelOutput(NamedTuple):
    representation: torch.Tensor


class MultiHead(torch.nn.Module):
    def __init__(self, args):
        super(MultiHead, self).__init__()

        from lib import nomenclature

        self.args = args

        if not isinstance(self.args.heads, list):
            self.args.heads = [self.args.heads]

        self.heads = torch.nn.ModuleDict({
            head_args.name: nomenclature.HEADS[head_args.kind](args = self.args, head_args = head_args.args)
            for head_args in self.args.heads
        })

    def forward(self, model_output: ModelOutput):
        aggregated_results = {}

        for name, module in self.heads.items():
            module_outputs = module(model_output)
            aggregated_results[name] = module_outputs

        return aggregated_results


class CoralHead(torch.nn.Module):
    def __init__(self, args, head_args = None):
        super(CoralHead, self).__init__()
        self.args = args
        self.head_args = head_args

        self.outputs = CoralLayer(
            size_in = self.args.model_args.latent_dim,
            num_classes = self.head_args.num_classes
        )

    def forward(self, model_output: ModelOutput) -> ClassificationOutput:
        logits = self.outputs(model_output.representation)
        probas = torch.sigmoid(logits)
        labels = proba_to_label(probas).float()

        output_results = ClassificationOutput(
            logits = logits,
            probas = probas,
            labels = labels
        )

        return output_results

class ClassificationHead(torch.nn.Module):
    def __init__(self, args, head_args = None):
        super(ClassificationHead, self).__init__()
        self.args = args
        self.head_args = head_args
        self.outputs = nn.Linear(self.args.model_args.latent_dim, self.head_args.num_classes, bias = False)

    def forward(self, model_output: ModelOutput) -> ClassificationOutput:
        logits = self.outputs(model_output.representation)
        probas = torch.nn.functional.softmax(logits, dim = -1)
        labels = logits.argmax(dim = -1)

        output_results = ClassificationOutput(
            logits = logits,
            probas = probas,
            labels = labels
        )

        return output_results

class MultiLabelHead(torch.nn.Module):
    def __init__(self, args, head_args = None):
        super(MultiLabelHead, self).__init__()
        self.args = args
        self.head_args = head_args
        self.outputs = nn.Linear(self.args.model_args.latent_dim, self.head_args.num_classes, bias = False)

    def forward(self, model_output: ModelOutput) -> ClassificationOutput:
        logits = self.outputs(model_output.representation)
        probas = torch.sigmoid(logits)
        labels = torch.round(probas)

        output_results = ClassificationOutput(
            logits = logits,
            probas = probas,
            labels = labels
        )

        return output_results
