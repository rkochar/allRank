import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE

from torch.nn import KLDivLoss, CrossEntropyLoss, MSELoss, L1Loss
from torch import rand

from allrank.models.losses.rankNet import rankNet
from allrank.models.losses.neuralNDCG import neuralNDCG
from allrank.models.losses.listMLE import listMLE
from allrank.models.losses.approxNDCG import approxNDCGLoss

def bestlossfunction(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    Pointwise RMSE loss.
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
    
    # CrossEntropy
    # crossloss = CrossEntropyLoss()
    # cl = crossloss(y_true, y_pred)
    
    # Kullback-Leibler Divergence Loss
    # leiblerloss = KLDivLoss()
    # ll = leiblerloss(y_true, y_pred)
    
    # RankNet
    # rn = rankNet(y_pred, y_true)
    
    # NeuralNDCG
    # nndcg = neuralNDCG(y_pred, y_true)
    
    # ApproxNDCG
    # andcg = approxNDCGLoss(y_pred, y_true)

    # Mean Squared Error
    # mseloss = MSELoss()
    # mse = mseloss(y_true, y_pred)

    # L1 loss
    L1loss = L1Loss()
    l1 = L1loss(y_true, y_pred)

    # Make your loss function here.
    # eg: mse + 4 * nndcg
    return l1
