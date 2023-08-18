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
        self.dataset = nomenclature.DATASETS[self.evaluator_args.dataset]

        self.val_dataloader = self.dataset.val_dataloader(args, kind = 'validation')
        self.device = device

    def trainer_evaluate(self, step):
        print("Running Evaluation.")
        results = self.evaluate(save=False)

        pprint.pprint(results)
        return results

    @torch.no_grad()
    def evaluate(self, save=True):
        y_preds = {}
        y_preds_proba = {}
        true_labels = {}

        y_preds_proba_over_time = defaultdict(lambda: {'preds': [], 'true_label': None})

        for i, batch in enumerate(tqdm(self.val_dataloader, total=len(self.val_dataloader))):
            finished = False
            current_latents = None

            for video_id, label in zip(batch['video_id'], batch['labels']):
                true_labels[video_id] = label.item()

            next_video_offsets = batch['next_window_offset'].cpu().numpy()

            progress_bars = {video_id: tqdm(range(total_windows), leave = False, colour="green") for video_id, total_windows in zip(batch['video_id'], batch['total_windows'])}

            while not finished:
                current_windows = {}

                # Slow as fuck
                for video_id, next_window_offset in zip(batch['video_id'], next_video_offsets):
                    new_sample = self.val_dataloader.dataset.get_batch(video_id, next_window_offset)

                    for key, value in new_sample.items():
                        if key not in current_windows:
                            current_windows[key] = []

                        current_windows[key].append(value)

                for key, value in current_windows.items():
                    current_windows[key] = np.stack(value, axis = 0)
                    if type(value[0]) == str:
                        continue

                    current_windows[key] = torch.from_numpy(current_windows[key]).to(self.device)

                    if 'modality' in key:
                        current_windows[key] = current_windows[key].squeeze(1)

                # update the things
                next_video_offsets = current_windows['next_window_offset'].cpu().numpy()

                model_output = self.model(current_windows, latent = current_latents)

                probas = model_output['depression'].probas[:, 1]
                current_latents = model_output['latent']

                finished_indexes = (current_windows['is_last'] == 1).detach().cpu().numpy()
                finished_video_ids = [batch['video_id'][idx] for idx in np.argwhere(finished_indexes).ravel()]
                not_finished_video_ids = [batch['video_id'][idx] for idx in np.argwhere(finished_indexes == False).ravel()]
                not_finished_offset_differences = current_windows['differences'][finished_indexes == False].detach().cpu().numpy()
                final_probas = probas[finished_indexes]

                for video_id, difference in zip(not_finished_video_ids, not_finished_offset_differences):
                    progress_bars[video_id].update(difference)

                for video_id, proba in zip(batch['video_id'], probas.cpu().numpy()):
                    if video_id in not_finished_video_ids:
                        y_preds_proba_over_time[video_id]['preds'].append(proba.item())
                        y_preds_proba_over_time[video_id]['true_label'] = true_labels[video_id]

                for video_id, proba in zip(finished_video_ids, final_probas):

                    if video_id in y_preds:
                        continue

                    y_preds[video_id] = proba.round().item()
                    y_preds_proba[video_id] = proba.item()

                    y_preds_proba_over_time[video_id]['preds'].append(proba.item()) # append the last one

                if torch.all(current_windows['is_last'] == 1):
                    finished = True

        sorted_keys = sorted(y_preds_proba_over_time.keys())

        y_preds_np = np.array([y_preds[key] for key in sorted_keys])
        y_preds_proba_np = np.array([y_preds_proba[key] for key in sorted_keys])
        true_labels_np = np.array([true_labels[key] for key in sorted_keys])

        fpr, tpr, thresholds = metrics.roc_curve(
            true_labels_np, y_preds_proba_np, pos_label=1
        )

        acc = metrics.accuracy_score(true_labels_np, y_preds_np)
        auc = metrics.auc(fpr, tpr)
        precision = metrics.precision_score(true_labels_np, y_preds_np)
        recall = metrics.recall_score(true_labels_np, y_preds_np)
        f1 = metrics.f1_score(true_labels_np, y_preds_np)

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
                f"results/{self.args.output_dir}/temporal-evaluator:{self.args.group}:{self.args.name}.csv",
                index=False,
            )

            with open(f"results/{self.args.output_dir}/temporal-evaluator:{self.args.group}:{self.args.name}:over-time.json", 'w') as f:
                json.dump(y_preds_proba_over_time, f, indent=4)

        return actual_results
