from .callback import Callback


class LambdaCallback(Callback):
    def __init__(self,
        on_epoch_end = None,
        on_epoch_start = None,
        on_batch_start = None,
        on_batch_end = None,
        on_training_start = None,
        on_training_end = None
    ):

        self._on_epoch_end = on_epoch_end
        self._on_epoch_start = on_epoch_start
        self._on_batch_start = on_batch_start
        self._on_batch_end = on_batch_end
        self._on_training_start = on_training_start
        self._on_training_end = on_training_end

    def on_epoch_end(self):
        if self._on_epoch_end is not None:
            self._on_epoch_end()

    def on_epoch_start(self):
        if self._on_epoch_start is not None:
            self._on_epoch_start()

    def on_batch_start(self):
        if self._on_batch_start is not None:
            self._on_batch_start()

    def on_batch_end(self):
        if self._on_batch_end is not None:
            self._on_batch_end()

    def on_training_start(self):
        if self._on_training_end is not None:
            self._on_training_end()

    def on_training_end(self):
        if self._on_training_end is not None:
            self._on_training_end()
