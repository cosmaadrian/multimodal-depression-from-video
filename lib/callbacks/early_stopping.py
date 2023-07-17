from .callback import Callback


class EarlyStopping(Callback):
    def __init__(self,
            monitor = 'val_loss',
            patience = 5,
            direction = 'down',
        ):
        self.monitor = monitor
        self.patience = patience
        self.direction = direction

        self.trainer = None

        self.current_patience = 0
        self.previous_best = None

    def on_epoch_end(self):
        trainer_quantity = self.trainer.logger.metrics[self.monitor]

        if self.previous_best is not None:
            if self.direction == 'down':
                if self.previous_best <= trainer_quantity:
                    self.current_patience += 1
                else:
                    self.current_patience = 0
                    self.previous_best = trainer_quantity
                    return
            elif self.direction == 'up':
                if self.previous_best >= trainer_quantity:
                    self.current_patience += 1
                else:
                    self.current_patience = 0
                    self.previous_best = trainer_quantity
                    return
        else:
            self.previous_best = trainer_quantity

        if self.current_patience == self.patience:
            print(f"No improvement for {self.patience} epochs. Stopping training.")
            self.trainer.stop()
