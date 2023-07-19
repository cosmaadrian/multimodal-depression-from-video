from lib.evaluator_extra import AcumenEvaluator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, zero_one_loss
import torch
import os
from tqdm import tqdm
import numpy as np
import json

class ClassificationEvaluator(AcumenEvaluator):
    def __init__(self, args, model, evaluator_args, logger = None):
        super(ClassificationEvaluator, self).__init__(args, model, logger = logger)
        from lib import nomenclature
        from lib import device

        self.evaluator_args = evaluator_args
        self.dataset = nomenclature.DATASETS[self.args.dataset]

        self.val_dataloader = self.dataset.val_dataloader(args, kind = 'validation')
        self.device = device

    def trainer_evaluate(self, step = None):
        return self.evaluate(save = False)

    @torch.no_grad()
    def evaluate(self, save = True):
        np.set_printoptions(suppress=True)
        y_pred = []
        y_true = []

        # TODO needs refactoring
        for i, batch in enumerate(tqdm(self.val_dataloader, total = len(self.val_dataloader))):
            for k, v in batch.items():
                batch[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v

            output = self.model(batch)['depression']

            labels = batch['labels'].detach().cpu().numpy().ravel().tolist()

            y_true.extend(labels)
            y_pred.extend(output.labels.detach().cpu().numpy().ravel().tolist())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        error = zero_one_loss(y_true, y_pred, normalize = True)

        results = {
            'acc': accuracy_score(y_true, y_pred),
            'prec': precision_score(y_true, y_pred, average = 'macro'),
            'recall': recall_score(y_true, y_pred, average = 'macro'),
            'f1': f1_score(y_true, y_pred, average = 'macro'),
            'error': error,
        }

        return results
