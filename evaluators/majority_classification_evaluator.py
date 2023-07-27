from lib.evaluator_extra import AcumenEvaluator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, zero_one_loss
import torch
import os
from tqdm import tqdm
import numpy as np
import json
from scipy import stats
from sklearn import metrics
import pandas as pd

# Majority Voting Evaluator

# Temporal Video Evaluator (take final decision after processing all the video)

class MajorityClassificationEvaluator(AcumenEvaluator):
    def __init__(self, args, model, evaluator_args, logger = None):
        super(MajorityClassificationEvaluator, self).__init__(args, model, logger = logger)
        from lib import nomenclature
        from lib import device

        self.evaluator_args = evaluator_args
        self.dataset = nomenclature.DATASETS[self.args.dataset]

        self.val_dataloader = self.dataset.val_dataloader(args, kind = 'validation')
        self.device = device

        self.num_runs = self.evaluator_args.num_eval_runs

    def trainer_evaluate(self, step):
        print("Running Evaluation.")
        results = self.evaluate(save=False, num_runs = 1)
        return results[-1]["f1"]

    def evaluate(self, save=True, num_runs = None):
        y_preds = []
        y_preds_proba = []
        true_labels = []

        if num_runs is None:
            num_runs = self.num_runs

        for _ in range(self.num_runs):
            y_pred = []
            y_pred_proba = []
            true_label = []

            with torch.no_grad():
                for i, batch in enumerate(tqdm(self.val_dataloader, total=len(self.val_dataloader))):
                    for key, value in batch.items():
                        batch[key] = value.to(self.nomenclature.device)

                    output = self.model(batch)["probas"]

                    preds = np.vstack(output.detach().cpu().numpy()).ravel()
                    labels = np.vstack(batch["labels"].detach().cpu().numpy()).ravel()

                    y_pred.extend(np.round(preds))
                    y_pred_proba.extend(preds)
                    true_label.extend(labels)

            y_preds.append(y_pred)
            y_preds_proba.append(y_pred_proba)
            true_labels.append(true_label)

        y_preds = np.array(y_preds)
        y_preds_proba = np.array(y_preds_proba)
        true_labels = np.array(true_labels)

        y_preds_voted = stats.mode(y_preds).mode[0]
        true_labels = stats.mode(true_labels).mode[0]
        y_preds_proba = y_preds_proba.mean(axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(
            true_labels, y_preds_proba, pos_label=1
        )
        acc = metrics.accuracy_score(true_labels, y_preds_voted)
        auc = metrics.auc(fpr, tpr)
        precision = metrics.precision_score(true_labels, y_preds_voted)
        recall = metrics.recall_score(true_labels, y_preds_voted)
        f1 = metrics.f1_score(true_labels, y_preds_voted)

        results = pd.DataFrame.from_dict(
            {
                "f1": [f1],
                "recall": [recall],
                "precision": [precision],
                "auc": [auc],
                "accuracy": [acc],
                "name": [f"{self.args.group}:{self.args.name}"],
                "dataset": [self.args.dataset],
                "model": [self.args.model],
            }
        )

        if save:
            results.to_csv(
                f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}.csv",
                index=False,
            )

        return results
