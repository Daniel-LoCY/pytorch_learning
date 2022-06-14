import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import torch
import matplotlib.pyplot as plt

x = torch.tensor([3.0])  # 蘋果單價
y = torch.tensor([18.0]) # 我們的預算
a = torch.tensor([1.0], requires_grad=True)  # 追蹤導數
print('grad:', a.grad)
loss = y - (a * x)  # loss function ( 中文稱 損失函數 )
loss.backward()
print('grad:', a.grad)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

for _ in range(100):
    a.grad.zero_()
    loss = y - (a * x)
    loss.backward()
    with torch.no_grad():
        a -= a.grad * 0.01 * loss
    writer.add_scalar('loss', loss, _)
