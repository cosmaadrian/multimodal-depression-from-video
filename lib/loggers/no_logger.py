class NoLogger(object):
    def __init__(self):
        self.on_step_metrics = dict()
        self.trainer = None
        self.metrics = dict()

    def watch(self, model):
        pass

    def log(self, key, value, on_step = True, force_log = False):
        self.metrics[key] = value

        if on_step:
            self.on_step_metrics[key] = value
