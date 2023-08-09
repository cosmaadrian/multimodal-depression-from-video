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
from collections import defaultdict

import pprint

class TemporalEvaluator(AcumenEvaluator):
    def __init__(self, args, model, evaluator_args, logger = None):
        super(TemporalEvaluator, self).__init__(args, model, logger = logger)
        from lib import nomenclature
        from lib import device

        self.nomenclature = nomenclature

        self.evaluator_args = evaluator_args
        self.dataset = nomenclature.DATASETS[self.args.dataset]

        self.val_dataloader = self.dataset.val_dataloader(args, kind = 'validation')
        self.device = device

        self.num_runs = self.evaluator_args.num_eval_runs

    def trainer_evaluate(self, step):
        print("Running Evaluation.")
        results = self.evaluate(save=False)

        pprint.pprint(results)

        return results

    @torch.no_grad()
    def evaluate(self, save=True, num_runs = None):
        y_preds = {}
        y_preds_proba = {}
        true_labels = {}

        y_preds_proba_over_time = defaultdict(list)

        for i, batch in enumerate(tqdm(self.val_dataloader, total=len(self.val_dataloader))):
            finished = False
            current_latents = None

            for video_id, label in zip(batch['video_id'], batch['labels']):
                true_labels[video_id] = label.item()

            while not finished:
                current_windows = {}

                # Slow as fuck
                for video_id, next_window_offset in zip(batch['video_id'], batch['next_window_offset']):
                    new_sample = self.val_dataloader.dataset.get_batch(video_id, next_window_offset)

                    for key, value in new_sample.items():
                        if key not in current_windows:
                            current_windows[key] = []

                        current_windows[key].append(value)

                for key, value in current_windows.items():
                    current_windows[key] = torch.cat(value, dim = 0).to(self.device)

                model_output = self.model(current_windows, latent = current_latents)

                probas = model_output['depression'].probas[:, 1]
                current_latents = model_output['latent']

                finished_video_ids = batch['video_id'][current_windows['is_last'] == 1]
                final_probas = probas[current_windows['is_last'] == 1]

                for video_id, proba in zip(finished_video_ids, final_probas):
                    y_preds_proba_over_time[video_id].append(proba.item())

                    if video_id in y_preds:
                        continue

                    y_preds[video_id] = proba.round().item()
                    y_preds_proba[video_id] = proba.item()

                if current_windows['is_last'].sum() == len(batch['video_id']):
                    finished = True

        exit()
        y_preds.append(y_pred)
        y_preds_proba.append(y_pred_proba)
        true_labels.append(true_label)

        y_preds = np.array(y_preds)
        y_preds_proba = np.array(y_preds_proba)
        true_labels = np.array(true_labels)

        y_preds_voted = stats.mode(y_preds, axis = 0).mode[0]
        true_labels = stats.mode(true_labels, axis = 0).mode[0]
        y_preds_proba = y_preds_proba.mean(axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(
            true_labels, y_preds_proba, pos_label=1
        )
        acc = metrics.accuracy_score(true_labels, y_preds_voted)
        auc = metrics.auc(fpr, tpr)
        precision = metrics.precision_score(true_labels, y_preds_voted)
        recall = metrics.recall_score(true_labels, y_preds_voted)
        f1 = metrics.f1_score(true_labels, y_preds_voted)

        results_for_logging = {
            "f1": [f1],
            "recall": [recall],
            "precision": [precision],
            "auc": [auc],
            "accuracy": [acc],
            "name": [f"{self.args.group}:{self.args.name}"],
            "dataset": [self.args.dataset],
            "model": [self.args.model],
        }

        actual_results = {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "auc": auc,
            "accuracy": acc,
        }

        if save:
            pd.DataFrame.from_dict(results_for_logging).to_csv(
                f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}.csv",
                index=False,
            )

        return actual_results
