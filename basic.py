import numpy as np
import torch
import matplotlib.pyplot as plt

# x = torch.tensor([3.0])  # 蘋果單價
# y = torch.tensor([18.0]) # 我們的預算
# a = torch.tensor([1.0], requires_grad=True)  # 追蹤導數
# print('grad:', a.grad)
# loss = y - (a * x)  # loss function ( 中文稱 損失函數 )
# loss.backward()
# print('grad:', a.grad)

# for _ in range(100):
#     a.grad.zero_()
#     loss = y - (a * x)
#     loss.backward()
#     with torch.no_grad():
#         a -= a.grad * 0.01 * loss
# print('a:', a)
# print('loss:', (y - (a * x)))
# print('result:', (a * x))

# x = torch.tensor([3.0, 5.0, 6.0,])   # 不同種蘋果售價
# y = torch.tensor([18.0, 18.0, 18.0]) # 我們的預算
# a = torch.tensor([1.0, 1.0, 1.0], requires_grad=True) # 先假設都只能買一顆
# loss_func = torch.nn.MSELoss()
# optimizer = torch.optim.SGD([a], lr=0.01)
# loss_list = []
# step = 100
# for _ in range(step):
#     optimizer.zero_grad()
#     loss = loss_func(y, a * x)
#     loss.backward()
#     loss_list.append(loss.detach().numpy())
#     optimizer.step()
#     print(a*x)
# print('a:', a)
# plt.plot(np.arange(1, step+1), loss_list)
# plt.show()


# from sklearn.datasets import make_regression
# np_x, np_y = make_regression(n_samples=500, n_features=10)
# x = torch.from_numpy(np_x).float()
# y = torch.from_numpy(np_y).float()

# w = torch.randn(10, requires_grad=True)
# b = torch.randn(1, requires_grad=True)
# optimizer = torch.optim.SGD([w, b], lr=0.01)
# def model(x):
#     return x @ w + b

# predict_y = model(x)
# print(predict_y)
# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.plot(y)
# plt.plot(predict_y.detach().numpy())
# plt.subplot(1, 2, 2)
# plt.scatter(y.detach().numpy(), predict_y.detach().numpy())
# plt.show()

# loss_func = torch.nn.MSELoss() # 之前提過的 loss function 
# history = []   # 紀錄 loss（誤差/損失）的變化
# for _ in range(300):   # 訓練 300 次
#     predict_y = model(x)
#     loss = loss_func(predict_y, y)
#     # 優化與 backward 動作，之前介紹過
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     history.append(loss.item())
# plt.plot(history)
# plt.show()
# predict_y = model(x)
# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.plot(y, label='target')
# plt.plot(predict_y.detach().numpy(), label='predict')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.scatter(y.detach().numpy(), predict_y.detach().numpy())
# plt.show()

# print(torch.cuda.is_available())