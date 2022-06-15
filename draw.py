from typing import BinaryIO
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
np_x, np_y = make_regression(n_samples=500, n_features=10)
x = torch.from_numpy(np_x).float()
y = torch.from_numpy(np_y).float()
w = torch.randn(10, requires_grad=True)
b = torch.randn(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.01)
def model(x):
    return x @ w + b

predict_y = model(x)
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(y)
plt.plot(predict_y.detach().numpy())
plt.subplot(1, 2, 2)
plt.scatter(y.detach().numpy(), predict_y.detach().numpy())
plt.show()

# writer = SummaryWriter()
# writer.add_figure('figure', plt.gcf())