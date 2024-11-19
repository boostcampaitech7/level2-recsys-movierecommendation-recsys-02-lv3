import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

'''
Loss Functions Collection
이 파일은 다양한 모델에서 사용되는 손실 함수(Loss Function)를 모아놓은 모듈입니다.
각 손실 함수는 PyTorch 기반으로 작성되었으며, 현재 VAE 모델의 손실 함수와
다른 모델들에서도 사용할 수 있는 다양한 손실 함수들이 포함될 예정입니다.

- MultiVAE_loss: Variational Autoencoder (VAE)에서 사용하는 손실 함수
- SASRec_loss :
-

이 모듈의 손실 함수들은 훈련 스크립트에서 불러와서 사용할 수 있도록 정의되어 있습니다.
'''


def multivae_loss(x, output, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(output[0], 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + output[2] - output[1].pow(2) - output[2].exp(), dim=1))

    return BCE + anneal * KLD



