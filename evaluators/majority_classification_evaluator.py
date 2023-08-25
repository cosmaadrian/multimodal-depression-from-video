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

# Majority Voting Evaluator

# Temporal Video Evaluator (take final decision after processing all the video)

class MajorityClassificationEvaluator(AcumenEvaluator):
    def __init__(self, args, model, evaluator_args, logger = None):
        super(MajorityClassificationEvaluator, self).__init__(args, model, logger = logger)
        from lib import nomenclature
        from lib import device

        self.nomenclature = nomenclature

        self.evaluator_args = evaluator_args
        self.dataset = nomenclature.DATASETS[self.evaluator_args.dataset]

        self.val_dataloader = self.dataset.val_dataloader(args, kind = self.evaluator_args.kind)
        self.device = device

        self.num_runs = self.evaluator_args.num_eval_runs

    def trainer_evaluate(self, step):
        print("Running Evaluation.")
        results = self.evaluate(save=False, num_runs = self.num_runs)

        pprint.pprint(results)

        return results

    def evaluate(self, save=True, num_runs = None):
        y_preds = []
        y_preds_proba = []
        true_labels = []

        y_preds_proba_over_runs = defaultdict(lambda: {'preds': [], 'true_label': None})

        if num_runs is None:
            num_runs = self.num_runs

        for _ in range(num_runs):
            y_pred = []
            y_pred_proba = []
            true_label = []

            with torch.no_grad():
                for i, batch in enumerate(tqdm(self.val_dataloader, total=len(self.val_dataloader), colour = 'green')):
                    latent = None
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)

                        if self.args.model == "baseline":
                            if 'modality' in key:
                                batch[key] = batch[key].squeeze(1)

                    if self.args.model == "perceiver":
                        for window_idx in range(0, self.args.n_temporal_windows):
                            window_batch = {}
                            window_batch['video_frame_rate'] = batch['video_frame_rate']
                            window_batch['audio_frame_rate'] = batch['audio_frame_rate']

                            for modality in self.args.modalities:
                                modality_id = modality.name
                                window_batch[f"modality:{modality_id}:data"] = batch[f"modality:{modality_id}:data"][:, window_idx, ...]
                                window_batch[f"modality:{modality_id}:mask"] = batch[f"modality:{modality_id}:mask"][:, window_idx, ...]

                            outputs = self.model(window_batch, latent = latent)
                            latent = outputs['latent']
                    else:
                        outputs = self.model(batch)
                        
                    output = outputs['depression'].probas[:, 1] # 0.0, 0.9

                    preds = np.vstack(output.detach().cpu().numpy()).ravel()
                    labels = np.vstack(batch["labels"].detach().cpu().numpy()).ravel()

                    y_pred.extend(np.round(preds))
                    y_pred_proba.extend(preds)
                    true_label.extend(labels)

                    for video_id, proba, the_true_label in zip(batch['video_id'], preds, labels):
                        y_preds_proba_over_runs[video_id]['preds'].append(proba.item())
                        y_preds_proba_over_runs[video_id]['true_label'] = the_true_label.item()

            y_preds.append(y_pred)
            y_preds_proba.append(y_pred_proba)
            true_labels.append(true_label)

        y_preds = np.array(y_preds)
        y_preds_proba = np.array(y_preds_proba)
        true_labels = np.array(true_labels)

        y_preds_voted = stats.mode(y_preds, axis = 0).mode[0]
        true_labels = stats.mode(true_labels, axis = 0).mode[0]
        y_preds_proba = y_preds_proba.mean(axis=0)

        print("True Labels:")
        print(true_labels)

        print("Voted Predictions:")
        print(y_preds_voted)

        print("Predictions Proba (Mean):")
        print(y_preds_proba)

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
            "dataset_kind": [self.evaluator_args.kind],
            "model": [self.args.model],
            "prediction_kind": [f'mean over {num_runs} runs']
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
                f"results/{self.args.output_dir}/majority-evaluator:{self.args.group}:{self.args.name}:{self.evaluator_args.kind}.csv",
                index=False,
            )

            with open(f"results/{self.args.output_dir}/majority-evaluator:{self.args.group}:{self.args.name}:over-runs:{self.evaluator_args.kind}.json", "w") as f:
                json.dump(y_preds_proba_over_runs, f, indent=4)

        return actual_results
