import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def loss(R, R_hat, U, V, lambda_reg=0.01):
    """
    EASE loss function with regularization.
    
    Parameters:
    - R: 실제 사용자-아이템 상호작용 행렬 (ground truth)
    - R_hat: 예측된 사용자-아이템 상호작용 행렬
    - U: 사용자 특징 행렬
    - V: 아이템 특징 행렬
    - lambda_reg: 정규화 계수
    
    Returns:
    - loss: 최종 손실값
    """
    # MSE Loss (실제 값과 예측값의 차이)
    mse_loss = np.mean((R - R_hat) ** 2)
    
    # Regularization (L2 regularization)
    reg_loss = lambda_reg * (np.sum(U ** 2) + np.sum(V ** 2))
    
    # Total Loss
    loss = mse_loss + reg_loss
    return loss