import wandb
from collections import defaultdict

class WandbLogger(object):
    def __init__(self):
        self.on_step_metrics = dict()
        self.trainer = None

        self.metrics = dict()

        self.min_values = defaultdict(lambda: 10000)
        self.max_values = defaultdict(lambda: -1)

    def watch(self, model):
        wandb.watch(model)

    def log_dict(self, log_dict, on_step = True, force_log = False):
        for key, value in log_dict.items():
            self.metrics[key] = value
            if on_step:
                self.on_step_metrics[key] = value

        if (self.trainer.global_step % self.trainer.args.log_every == 0) or force_log:
            if wandb.run is not None:
                wandb.log(log_dict, step = self.trainer.global_step)

    def log(self, key, value, on_step = True, force_log = False, log_extremes = False):
        self.metrics[key] = value

        if on_step:
            self.on_step_metrics[key] = value

        if (self.trainer.global_step % self.trainer.args.log_every == 0) or force_log:
            if wandb.run is not None:
                log_dict = {
                    key: value
                }

                if log_extremes:
                    self.min_values[key] = min(self.min_values[key], value)
                    self.max_values[key] = max(self.max_values[key], value)

                    log_dict[key + ':min'] = self.min_values[key]
                    log_dict[key + ':max'] = self.max_values[key]

                wandb.log(log_dict, step = self.trainer.global_step)
