import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def deepfm_loss(y_pred, y_true):
    """
    DeepFM 모델을 위한 손실 함수 (이진 교차 엔트로피).

    parameters
    ----------
    y_true: 실제값 (배치 크기 x 1)
    y_pred: 예측값 (배치 크기 x 1)

    Returns
    -------
    loss: 계산된 손실값
    """
    # Binary Cross Entropy (BCE) loss
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(y_pred, y_true)

    return loss
