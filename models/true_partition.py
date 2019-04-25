from models.base_model import BaseModel


class TruePartitionModel(BaseModel):
    def __init__(self, true_labels=None):
        self.true_labels = true_labels

    def fit_transform(self, graph):
        return self.true_labels
