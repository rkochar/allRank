import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE

from torch.nn import L1Loss
from torch import rand

from allrank.models.losses.neuralNDCG import neuralNDCG
import math

x = 30*11

def iterNN(y_pred, y_true, epoch, padded_value_indicator=PADDED_Y_VALUE, temperature=1., powered_relevancies=True, k=None,
           stochastic=False, n_samples=32, beta=0.1, log_scores=True):
    """
    Iteration Sensitive NeuralNDCG.

    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """

    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    valid_mask = (y_true != padded_value_indicator).type(torch.float32)

    y_true[mask] = 0
    y_pred[mask] = 0

    # L1 loss
    L1loss = L1Loss()

    # Calculate weight and return iterNN
    weight = math.exp(-epoch * 0.02)
    return (weight * (L1loss(y_true, y_pred)) + (1 - weight) * neuralNDCG(y_pred, y_true,padded_value_indicator=padded_value_indicator))
