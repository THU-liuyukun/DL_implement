import time
import torch
import torch.nn.functional as F
from torch import nn

# 输入2 * 9 * 9的图片
in_channels = 2
out_channels = 2
kernel_size = 3
w = 9
h = 9

# input
x = torch.ones(1, in_channels, w, h)

# 方法1：原生实现
# 公式：res_block = 3 * 3 conv + 1 * 1 conv + input
conv_2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
conv_2d_point_wise = nn.Conv2d(in_channels, out_channels, 1)

result1 = conv_2d(x) + conv_2d_point_wise(x) + x

# 方法2：算子融合
# 1、改造
# point-wise卷积 -> 3 * 3
# [2, 2, 1, 1] -> [2, 2, 3, 3]
point_wise_to_conv_weight = F.pad(conv_2d_point_wise.weight, [1, 1, 1, 1, 0, 0, 0, 0])
conv_2d_for_point_wise = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
# weight是parameter，将张量转换成parameter
conv_2d_for_point_wise.weight = nn.Parameter(point_wise_to_conv_weight)
conv_2d_for_point_wise.bias = conv_2d_point_wise.bias

# x -> 3 * 3
zeros = torch.unsqueeze(torch.zeros(kernel_size, kernel_size), 0)
stars = torch.unsqueeze(F.pad(torch.ones(1, 1), [1, 1, 1, 1]), 0)
stars_zeros = torch.unsqueeze(torch.cat([stars, zeros], 0), 0)
zeros_stars = torch.unsqueeze(torch.cat([zeros, stars], 0), 0)
identity_to_conv_weight = torch.cat([stars_zeros, zeros_stars], 0)
identity_to_conv_bias = torch.zeros([out_channels])
conv_2d_for_identity = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
conv_2d_for_identity.weight = nn.Parameter(identity_to_conv_weight)
conv_2d_for_identity.bias = nn.Parameter(identity_to_conv_bias)

result2 = conv_2d(x) + conv_2d_for_point_wise(x) + conv_2d_for_identity(x)
print(torch.all(torch.isclose(result1, result2)))

# 2、融合
conv_2d_for_fusion = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
conv_2d_for_fusion.weight = nn.Parameter(conv_2d.weight.data + conv_2d_for_point_wise.weight.data + conv_2d_for_identity.weight.data)
conv_2d_for_fusion.bias = nn.Parameter(conv_2d.bias.data + conv_2d_for_point_wise.bias.data + conv_2d_for_identity.bias.data)
result3 = conv_2d_for_fusion(x)
print(torch.all(torch.isclose(result2, result3)))

# 原生写法和算子融合的效率对比
t1 = time.time()
conv_2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
conv_2d_point_wise = nn.Conv2d(in_channels, out_channels, 1)
result1 = conv_2d(x) + conv_2d_point_wise(x) + x
t2 = time.time()

t3 = time.time()
conv_2d_for_fusion = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
conv_2d_for_fusion.weight = nn.Parameter(conv_2d.weight.data + conv_2d_for_point_wise.weight.data + conv_2d_for_identity.weight.data)
conv_2d_for_fusion.bias = nn.Parameter(conv_2d.bias.data + conv_2d_for_point_wise.bias.data + conv_2d_for_identity.bias.data)
result3 = conv_2d_for_fusion(x)
t4 = time.time()
print("原生写法耗时：", (t2 - t1) * 1000000)
print("算子融合写法耗时：", (t4 - t3) * 1000000)
