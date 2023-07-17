

class Modality(object):
    def __init__(self, args):
        self.args = args

    def read_chunk(self, window):
        raise Exception('Not Implemented')

    def post_process(self, data):
        # TODO add padding and stuff like that
        return data
