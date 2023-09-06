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
    
    def compute_metrics(self, y_labels, y_preds_proba, y_preds):
        metrics_results = {}

        fpr, tpr, thresholds = metrics.roc_curve(
            y_labels, y_preds_proba, pos_label=1
        )

        metrics_results["fpr"] = fpr
        metrics_results["tpr"] = tpr
        metrics_results["thresholds"] = thresholds
        metrics_results["acc"] = metrics.accuracy_score(y_labels, y_preds)
        metrics_results["auc"] = metrics.auc(fpr, tpr)
        metrics_results["precision"] = metrics.precision_score(y_labels, y_preds)
        metrics_results["recall"] = metrics.recall_score(y_labels, y_preds)
        metrics_results["f1"] = metrics.f1_score(y_labels, y_preds)
        metrics_results["f1_weighted"] = metrics.f1_score(y_labels, y_preds, average = "weighted")

        return metrics_results

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

                # getting the current window input data
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

                # to get the next window input data
                next_video_offsets = current_windows['next_window_offset'].cpu().numpy()

                # getting predictions for the current window
                model_output = self.model(current_windows, latent = current_latents)

                probas = model_output['depression'].probas[:, 1]
                current_latents = model_output['latent']

                # checking already processed videos + progress bar implementation
                finished_indexes = (current_windows['is_last'] == 1).detach().cpu().numpy()
                finished_video_ids = [batch['video_id'][idx] for idx in np.argwhere(finished_indexes).ravel()]
                not_finished_video_ids = [batch['video_id'][idx] for idx in np.argwhere(finished_indexes == False).ravel()]
                not_finished_offset_differences = current_windows['differences'][finished_indexes == False].detach().cpu().numpy()
                final_probas = probas[finished_indexes]
                final_satisty_presence_thr = current_windows["satisfy_presence_thr"][finished_indexes]

                # progress bar updating
                for video_id, difference in zip(not_finished_video_ids, not_finished_offset_differences):
                    progress_bars[video_id].update(difference)

                # gathering predictions over the time
                for video_id, proba, satisfy_presence_thr in zip(batch['video_id'], probas.cpu().numpy(), current_windows["satisfy_presence_thr"]):
                    if video_id in not_finished_video_ids:
                        if satisfy_presence_thr:
                            y_preds_proba_over_time[video_id]['preds'].append(proba.item())

                            if proba.item() > self.evaluator_args.max_threshold or proba.item() < self.evaluator_args.min_threshold:
                                y_preds_proba_over_time[video_id]['preds_threshold'].append(proba.item())

                            y_preds_proba_over_time[video_id]['true_label'] = true_labels[video_id]

                            # just in case the last window do not satisfy the presence threshold condition
                            # we take the last window probability that actually satisfied that condition
                            y_preds[video_id] = proba.round().item()
                            y_preds_proba[video_id] = proba.item()

                # taking the prediction of the last window once the video has been entirely processed
                for video_id, proba, satisfy_presence_thr in zip(finished_video_ids, final_probas, final_satisty_presence_thr):
                    if video_id in y_preds:
                        continue
                    if not satisfy_presence_thr:
                        continue

                    y_preds[video_id] = proba.round().item()
                    y_preds_proba[video_id] = proba.item()
                    y_preds_proba_over_time[video_id]['preds'].append(proba.item()) # append the last one

                    # get preds over threshold less than min_threshold or greater than max_threshold
                    if proba.item() > self.evaluator_args.max_threshold or proba.item() < self.evaluator_args.min_threshold:
                        y_preds_proba_over_time[video_id]['preds_threshold'].append(proba.item())

                # while loop break condition checking
                if torch.all(current_windows['is_last'] == 1):
                    finished = True


        #######################
        ## COMPUTING METRICS ##
        #######################

        # sanity ordering
        sorted_keys = sorted(y_preds_proba_over_time.keys())

        true_labels_np = np.array([true_labels[key] for key in sorted_keys])

        # metrics w.r.t. the last window
        y_preds_np = np.array([y_preds[key] for key in sorted_keys])
        y_preds_proba_np = np.array([y_preds_proba[key] for key in sorted_keys])

        metrics_last = self.compute_metrics(true_labels_np, y_preds_proba_np, y_preds_np)


        # metrics but as a mean of predictions over time
        y_preds_proba_mean_over_time_np = np.array([np.mean(y_preds_proba_over_time[key]['preds']) for key in sorted_keys])
        y_preds_mean_over_time_np = y_preds_proba_mean_over_time_np.round()

        metrics_mean_over_time = self.compute_metrics(true_labels_np, y_preds_proba_mean_over_time_np, y_preds_mean_over_time_np)

        if self.evaluator_args.kind == "validation":
            # computing optimum threshold
            gmeans = np.sqrt(metrics_mean_over_time["tpr"] * (1-metrics_mean_over_time["fpr"]))
            opt_thr_idx = np.argmax(gmeans)
            opt_thr = metrics_mean_over_time["thresholds"][opt_thr_idx]
            print(f"\nOPTIMUM THRESHOLD: {opt_thr}\n")

        # metrics but as a mode of predictions over time
        y_preds_mode_over_time_np = np.array([stats.mode(np.array(y_preds_proba_over_time[key]['preds']).round())[0].item() for key in sorted_keys])

        metrics_mode_over_time = self.compute_metrics(true_labels_np, y_preds_proba_mean_over_time_np, y_preds_mode_over_time_np)


        # metrics but using preds_threshold
        y_preds_proba_over_time_threshold = np.array([np.mean(y_preds_proba_over_time[key]['preds_threshold'])
            if len(y_preds_proba_over_time[key]['preds_threshold']) > 0 else np.mean(y_preds_proba_over_time[key]['preds'])
            for key in sorted_keys])
        y_preds_over_time_np_threshold = y_preds_proba_over_time_threshold.round()

        metrics_threshold = self.compute_metrics(true_labels_np, y_preds_proba_over_time_threshold, y_preds_over_time_np_threshold)

        # metrics but using preds_threshold + mode instead of mean
        y_preds_mode_over_time_np_threshold = np.array([stats.mode(np.array(y_preds_proba_over_time[key]['preds_threshold']).round())[0].item()
            if len(y_preds_proba_over_time[key]['preds_threshold']) > 0 else stats.mode(np.array(y_preds_proba_over_time[key]['preds']).round())[0].item()
            for key in sorted_keys])

        metrics_mode_threshold = self.compute_metrics(true_labels_np, y_preds_proba_over_time_threshold, y_preds_mode_over_time_np_threshold)


        results_for_logging = {
            "name": [f"{self.args.group}:{self.args.name}"] * 5,
            "run_id": [self.args.run_id] * 5,
            "f1": [metrics_last["f1"], metrics_mean_over_time["f1"], metrics_mode_over_time["f1"], metrics_threshold["f1"], metrics_mode_threshold["f1"]],
            "f1_weighted": [metrics_last["f1_weighted"], metrics_mean_over_time["f1_weighted"], metrics_mode_over_time["f1_weighted"], metrics_threshold["f1_weighted"], metrics_mode_threshold["f1_weighted"]],
            "recall": [metrics_last["recall"], metrics_mean_over_time["recall"], metrics_mode_over_time["recall"], metrics_threshold["recall"], metrics_mode_threshold["recall"]],
            "precision": [metrics_last["precision"], metrics_mean_over_time["precision"], metrics_mode_over_time["precision"], metrics_threshold["precision"], metrics_mode_threshold["precision"]],
            "auc": [metrics_last["auc"], metrics_mean_over_time["auc"], metrics_mode_over_time["auc"], metrics_threshold["auc"], metrics_mode_threshold["auc"]],
            "accuracy": [metrics_last["acc"], metrics_mean_over_time["acc"], metrics_mode_over_time["acc"], metrics_threshold["acc"], metrics_mode_threshold["acc"]],
            "dataset": [self.args.dataset] * 5,
            "dataset_kind": [self.evaluator_args.kind] * 5,
            "model": [self.args.model] * 5,
            "seconds_per_window": [self.args.seconds_per_window] * 5,
            "presence_threshold": [self.args.presence_threshold] * 5,
            "modalities": [self.args.use_modalities] * 5,
            "model_args.num_layers": [self.args.model_args.num_layers] * 5,
            "model_args.self_attn_num_heads": [self.args.model_args.self_attn_num_heads] * 5,
            "model_args.self_attn_dim_head": [self.args.model_args.self_attn_dim_head] * 5,
            "prediction_kind": ['last', 'mean', 'mode', 'threshold', 'mode_threshold']
        }

        actual_results = {
            "f1": metrics_mean_over_time["f1"],
            "recall": metrics_mean_over_time["recall"],
            "precision": metrics_mean_over_time["precision"],
            "auc": metrics_mean_over_time["auc"],
            "accuracy": metrics_mean_over_time["acc"],
        }

        if save:
            pd.DataFrame.from_dict(results_for_logging).to_csv(
                f"results/{self.args.output_dir}/temporal-evaluator:{self.args.group}:{self.args.name}:{self.evaluator_args.kind}.csv",
                index=False,
            )

            with open(f"results/{self.args.output_dir}/temporal-evaluator:{self.args.group}:{self.args.name}:over-time:{self.evaluator_args.kind}.json", 'w') as f:
                json.dump(y_preds_proba_over_time, f, indent=4)

        return actual_results
