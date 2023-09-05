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

        self.val_dataloader = self.dataset.val_dataloader(args, kind = self.evaluator_args.kind)
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

        y_preds_proba_over_time = defaultdict(lambda: {'preds': [], 'preds_threshold': [], 'true_label': None})

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

                        if proba.item() > self.evaluator_args.max_threshold or proba.item() < self.evaluator_args.min_threshold:
                            y_preds_proba_over_time[video_id]['preds_threshold'].append(proba.item())

                        y_preds_proba_over_time[video_id]['true_label'] = true_labels[video_id]

                for video_id, proba in zip(finished_video_ids, final_probas):

                    if video_id in y_preds:
                        continue

                    y_preds[video_id] = proba.round().item()
                    y_preds_proba[video_id] = proba.item()

                    y_preds_proba_over_time[video_id]['preds'].append(proba.item()) # append the last one

                    # get preds over threshold less than min_threshold or greater than max_threshold
                    if proba.item() > self.evaluator_args.max_threshold or proba.item() < self.evaluator_args.min_threshold:
                        y_preds_proba_over_time[video_id]['preds_threshold'].append(proba.item())

                if torch.all(current_windows['is_last'] == 1):
                    finished = True

        sorted_keys = sorted(y_preds_proba_over_time.keys())

        y_preds_np = np.array([y_preds[key] for key in sorted_keys])
        y_preds_proba_np = np.array([y_preds_proba[key] for key in sorted_keys])
        true_labels_np = np.array([true_labels[key] for key in sorted_keys])

        fpr, tpr, thresholds = metrics.roc_curve(
            true_labels_np, y_preds_proba_np, pos_label=1
        )

        # computing optimum threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        opt_thr_idx = np.argmax(gmeans)
        opt_thr = thresholds[opt_thr_idx]
        print(f"\nOPTIMUM THRESHOLD: {opt_thr}\n")

        acc = metrics.accuracy_score(true_labels_np, y_preds_np)
        auc = metrics.auc(fpr, tpr)
        precision = metrics.precision_score(true_labels_np, y_preds_np)
        recall = metrics.recall_score(true_labels_np, y_preds_np)
        f1 = metrics.f1_score(true_labels_np, y_preds_np)

        #### compute the same metrics but as a mean of predictions over time
        y_preds_proba_over_time_np = np.array([np.mean(y_preds_proba_over_time[key]['preds']) for key in sorted_keys])
        y_preds_over_time_np = y_preds_proba_over_time_np.round()

        fpr_over_time, tpr_over_time, thresholds_over_time = metrics.roc_curve(
            true_labels_np, y_preds_proba_over_time_np, pos_label=1
        )

        acc_over_time = metrics.accuracy_score(true_labels_np, y_preds_over_time_np)
        auc_over_time = metrics.auc(fpr_over_time, tpr_over_time)
        precision_over_time = metrics.precision_score(true_labels_np, y_preds_over_time_np)
        recall_over_time = metrics.recall_score(true_labels_np, y_preds_over_time_np)
        f1_over_time = metrics.f1_score(true_labels_np, y_preds_over_time_np)

        #### compute the same metrics but using preds_threshold

        y_preds_proba_over_time_threshold = np.array([np.mean(y_preds_proba_over_time[key]['preds_threshold'])
            if len(y_preds_proba_over_time[key]['preds_threshold']) > 0 else np.mean(y_preds_proba_over_time[key]['preds'])
            for key in sorted_keys])

        empty_keys = [key for key in sorted_keys if len(y_preds_proba_over_time[key]['preds_threshold']) == 0]

        y_preds_over_time_np_threshold = y_preds_proba_over_time_threshold.round()


        fpr_threshold, tpr_threshold, _ = metrics.roc_curve(
            true_labels_np, y_preds_over_time_np_threshold, pos_label=1
            )

        acc_threshold = metrics.accuracy_score(true_labels_np, y_preds_over_time_np_threshold)
        auc_threshold = metrics.auc(fpr_threshold, tpr_threshold)
        precision_threshold = metrics.precision_score(true_labels_np, y_preds_over_time_np_threshold)
        recall_threshold = metrics.recall_score(true_labels_np, y_preds_over_time_np_threshold)
        f1_threshold = metrics.f1_score(true_labels_np, y_preds_over_time_np_threshold)

        for key in sorted_keys:
            y_preds_proba_over_time['preds_mean'] = np.mean(y_preds_proba_over_time[key]['preds'])
            if len(y_preds_proba_over_time[key]['preds_threshold']) > 0:
                y_preds_proba_over_time['preds_threshold_mean'] = np.mean(y_preds_proba_over_time[key]['preds_threshold'])
            else:
                y_preds_proba_over_time['preds_threshold_mean'] = np.mean(y_preds_proba_over_time[key]['preds'])


        results_for_logging = {
            "name": [f"{self.args.group}:{self.args.name}"] * 3,
            "run_id": [self.args.run_id] * 3,
            "f1": [f1, f1_over_time, f1_threshold],
            "recall": [recall, recall_over_time, recall_threshold],
            "precision": [precision, precision_over_time, precision_threshold],
            "auc": [auc, auc_over_time, auc_threshold],
            "accuracy": [acc, acc_over_time, acc_threshold],
            "dataset": [self.args.dataset] * 3,
            "dataset_kind": [self.evaluator_args.kind] * 3,
            "model": [self.args.model] * 3,
            "seconds_per_window": [self.args.seconds_per_window] * 3,
            "presence_threshold": [self.args.presence_threshold] * 3,
            "modalities": [self.args.use_modalities] * 3,
            "model_args.num_layers": [self.args.model_args.num_layers] * 3,
            "model_args.self_attn_num_heads": [self.args.model_args.self_attn_num_heads] * 3,
            "model_args.self_attn_dim_head": [self.args.model_args.self_attn_dim_head] * 3,
            "prediction_kind": ['last', 'mean', 'threshold']
        }

        actual_results = {
            "f1": f1_over_time,
            "recall": recall_over_time,
            "precision": precision_over_time,
            "auc": auc_over_time,
            "accuracy": acc_over_time,
        }

        if save:
            pd.DataFrame.from_dict(results_for_logging).to_csv(
                f"results/{self.args.output_dir}/temporal-evaluator:{self.args.group}:{self.args.name}:{self.evaluator_args.kind}.csv",
                index=False,
            )

            with open(f"results/{self.args.output_dir}/temporal-evaluator:{self.args.group}:{self.args.name}:over-time:{self.evaluator_args.kind}.json", 'w') as f:
                json.dump(y_preds_proba_over_time, f, indent=4)

        return actual_results
