# r-dropout的粗略实现

import torch
import numpy as np
from torch import nn


def train(ratio, x, w1, b1, w2, b2):
    x = torch.cat([x, x], 0)
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    mask1 = np.random.binomial(1, 1 - ratio, layer1.shape)
    layer1 = layer1 * mask1

    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    mask2 = np.random.binomial(1, 1 - ratio, layer2.shape)
    layer2 = layer2 * mask2

    logits = func(layer2)
    # bs 是 batch size
    logits1, logits2 = logits[:bs, :], logits[bs:, :]
    nll = nn.NLLLoss()
    nll1 = nll(logits1, label)
    nll2 = nll(logits2, label)
    kl_loss = kl(logits1, logits2)
    loss = nll1 + nll2 + kl_loss

    return loss
