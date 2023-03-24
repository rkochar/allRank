import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from torch.nn import KLDivLoss, HingeEmbeddingLoss, CrossEntropyLoss, BCEWithLogitsLoss, MarginRankingLoss
from torch import rand

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

    crossloss = CrossEntropyLoss()
    cl = crossloss(y_true, y_pred)
    print("cl: " + str(cl))

    leiblerloss = KLDivLoss()
    ll = leiblerloss(y_true, y_pred)
    print(type(ll))
    print("ll: " + str(ll))
    print("lltest: " + str(ll[1]))

    # marginrankingloss = MarginRankingLoss()
    # mrl = marginrankingloss(y_true, y_pred)
    # print(mrl)

    return ll