import torch
from lib.model_extra import MultiHead, ModelOutput

class PerceiverModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.output = MultiHead(args)

    def forward(self, batch):
        return torch.zeros(16)
        # TODO ...
        # output = self.model(batch)

        # model_output = ModelOutput(representation = )
        # model_output = self.output(model_output)

        # return model_output

