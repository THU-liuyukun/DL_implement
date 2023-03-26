import numpy as np

def train1(ratio, x, w1, b1, w2, b2):
    # relu + linear
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    # 二项分布生成mask矩阵
    mask1 = np.random.binomial(1, 1 - ratio, layer1.shape)
    layer1 = layer1 * mask1

    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    mask2 = np.random.binomial(1, 1 - ratio, layer2.shape)
    layer2 = layer2 * mask2

    return layer2

def test1(ratio, x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    layer1 = layer1 * (1 - ratio)

    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    layer2 = layer2 * (1 - ratio)

    return layer2

# 训练时进行scale操作，减少inference计算量
def train2(ratio, x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    mask1 = np.random.binomial(1, 1 - ratio, layer1.shape)
    layer1 = layer1 * mask1
    layer1 = layer1 / (1 - ratio)

    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    mask2 = np.random.binomial(1, 1 - ratio, layer2.shape)
    layer2 = layer2 * mask2
    layer2 = layer2 / (1 - ratio)

    return layer2

def test2(x, w1, b1, w2, b2):
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)

    return layer2
