import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from torch.nn import KLDivLoss, HingeEmbeddingLoss, CrossEntropyLoss, BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, NLLLoss2d, MSELoss, L1Loss
from torch import rand
from allrank.models.losses.rankNet import rankNet
from allrank.models.losses.neuralNDCG import neuralNDCG
from allrank.models.losses.listMLE import listMLE
from allrank.models.losses.approxNDCG import approxNDCGLoss

def bestlossfunction(y_pred, y_true,epoch, padded_value_indicator=PADDED_Y_VALUE):
    """
    Pointwise RMSE loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """

    #print('epoch running',epoch)
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    valid_mask = (y_true != padded_value_indicator).type(torch.float32)

    y_true[mask] = 0
    y_pred[mask] = 0

    # crossloss = CrossEntropyLoss()
    # cl = crossloss(y_true, y_pred)
    # print("cl: " + str(cl))

    #leiblerloss = KLDivLoss()
    #ll = leiblerloss(y_true, y_pred)
    # print("ll: " + str(ll))

    # rn = rankNet(y_pred, y_true)
    # nndcg = neuralNDCG(y_pred, y_true)
    #andcg = approxNDCGLoss(y_pred, y_true)

    # mle = listMLE(y_true, y_pred)

    # marginrankingloss = MarginRankingLoss()
    # mrl = marginrankingloss(y_true, y_pred)
    # grad

    # hinge = HingeEmbeddingLoss()
    # hingeloss = hinge(y_true, y_pred)
    # grad

    # nll = NLLLoss2d()
    # nl = nll(y_true, y_pred)
    # dim

    mseloss = MSELoss()
    mse = mseloss(y_true, y_pred)

    # L1loss = L1Loss()
    # l1 = L1loss(y_true, y_pred)

    return  mse
