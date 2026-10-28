"""Perceptron multicouche peu profond (étape 4.1 du planning)."""

from torch import nn


class MLP(nn.Module):
    """Couches cachées entièrement connectées (ReLU puis dropout) et une
    couche de sortie linéaire qui produit un logit par classe.

    Pour le classifieur binaire, les logits sont appris avec
    CrossEntropyLoss et convertis en probabilités par softmax à l'inférence.
    """

    def __init__(self, n_inputs, hidden_sizes, n_outputs, dropout):
        super().__init__()

        layers = []
        size = n_inputs
        for hidden_size in hidden_sizes:
            layers += [
                nn.Linear(size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            size = hidden_size
        layers.append(nn.Linear(size, n_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
