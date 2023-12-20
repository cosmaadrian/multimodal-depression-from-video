import torch
from torch import nn
from torchinfo import summary

import numpy as np
import tqdm

from .loggers import NoLogger
import lib
import constants

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()

class NotALightningTrainer():

    def __init__(self,
            args,
            callbacks = None,
            logger = None
        ):
        self.args = args

        self.epoch = 0
        self.global_step = 0

        self.logger = logger

        if self.logger is None:
            self.logger = NoLogger()

        self.logger.trainer = self

        self.callbacks = callbacks
        if self.callbacks is None:
            self.callbacks = []

        for callback in self.callbacks:
            callback.trainer = self

        self.should_stop = False
        self.model_hook = None
        self.scaler = None


    def stop(self):
        self.should_stop = True

    def fit(self, model, train_dataloader, evaluators = None):
        model.log = self.logger.log

        if evaluators is None:
            evaluators = []

        for evaluator in evaluators:
            evaluator.trainer = self

        self.optimizer = model.configure_optimizers()
        model.trainer = self
        self.model_hook = model.model

        if not hasattr(self.model_hook, 'module'):
            model.model = nn.DataParallel(model.model)
            model.model = model.model.to(lib.device)
        try:
            summary(self.model_hook)
        except Exception as e:
            print("::: ⚠️WARNING⚠️ could not create model summary ::: ", e)

        self.logger.watch(self.model_hook)

        self.scaler = torch.cuda.amp.GradScaler(enabled = bool(self.args.use_amp))

        model.training_start()

        for epoch in range(self.args.epochs):
            if self.should_stop:
                break

            for callback in self.callbacks:
                callback.on_epoch_start()

            pbar = tqdm.tqdm(train_dataloader, total = len(train_dataloader), colour = 'cyan')

            model.training_epoch_start(epoch)
            for i, data in enumerate(pbar):
                self.global_step += 1

                for callback in self.callbacks:
                    callback.on_batch_start()

                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(lib.device)

                with torch.cuda.amp.autocast(enabled = bool(self.args.use_amp)):
                    loss = model.training_step(data, i)
                    loss = loss / self.args.accumulation_steps

                self.scaler.scale(loss).backward()

                if (i + 1) % self.args.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(self.model_hook.parameters(), 1.5)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.optimizer.zero_grad(set_to_none = True)

                    for callback in self.callbacks:
                        callback.on_batch_end()

                    model.training_batch_end()

                progress_string = f'[{Fore.GREEN}{self.args.group}{Style.RESET_ALL}:{Fore.RED}{self.args.name}{Style.RESET_ALL}] ' + \
                    f'Epoch {self.epoch} / {self.args.epochs} | ' + ' | '.join([
                    f'{k}={np.round(v, 4)}' for k,v in self.logger.on_step_metrics.items()
                ])

                pbar.set_description(progress_string)
                if self.args.debug:
                    print("[🐞DEBUG MODE🐞] Breaking after one batch ... ")
                    break

            model.training_epoch_end(epoch)
            self.epoch += 1
            self.logger.log('epoch', self.epoch, on_step = False, force_log = True)

            if (self.epoch + 1) % self.args.eval_every == 0:
                self.model_hook.train(False)
                with torch.no_grad():
                    for evaluator in evaluators:
                        print(f'[{evaluator.__class__.__name__}] Running evaluation ...')
                        values = evaluator.trainer_evaluate(self.global_step)
                        for key, value in values.items():
                            self.logger.log(f'{evaluator.__class__.__name__}_{key}', value, on_step = False, force_log = True, log_extremes = True)

                self.model_hook.train(True)

            for callback in self.callbacks:
                callback.on_epoch_end()

        model.training_end()
