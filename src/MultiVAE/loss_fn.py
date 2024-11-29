import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def multivae_loss(x, output, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(output[0], 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + output[2] - output[1].pow(2) - output[2].exp(), dim=1))

    return BCE + anneal * KLD



