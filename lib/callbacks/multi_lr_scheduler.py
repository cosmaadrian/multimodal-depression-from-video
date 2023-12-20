from .callback import Callback


class MultiLRSchedule(Callback):

    def __init__(self, schedulers):
        print(schedulers)
        self.schedulers = [sched for (sched, _, _) in schedulers]
        self.intervals = [(start, end) for (_, start, end) in schedulers]

        self.trainer = None
        self.current_scheduler = self.schedulers[0]

    def get_last_lr(self):
        return self.current_scheduler.get_last_lr()

    def on_batch_end(self):
        self.current_scheduler.step()

    def on_epoch_end(self):
        for i, (start, end) in enumerate(self.intervals):
            if self.trainer.epoch >= start and self.trainer.epoch < end:
                self.current_scheduler = self.schedulers[i]
                break
