import torch
import torch.nn.functional as F


class SimCLR(torch.nn.Module):
    def __init__(self, args):
        super(SimCLR, self).__init__()
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size // 2) for i in range(self.args.num_views)], dim=0)
        labels = labels.sort().values
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # labels = labels.to(features.device)

        features = F.normalize(features, dim = 1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)

        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / self.args.loss_args['temperature']

        loss = self.criterion(logits, labels)

        return loss