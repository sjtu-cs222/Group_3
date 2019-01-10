import torch
import Model
import DataPreprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas


learningRate = 0.05
epoch = 10000

net = Model.BasicNetwork(60,100,7)

optimizer = torch.optim.SGD(net.parameters(), learningRate)
loss_func = torch.nn.MSELoss()

csv_data, ave1 = DataPreprocess.open_1()

# show
x_axis = []
y_axis = []

for t in range(epoch):
    x = torch.Tensor(60).zero_()
    real = torch.Tensor(7).zero_()

    line = DataPreprocess.read_1(csv_data, ave1, t)

    for i in range(60):
        x[i] = line[i]
    for i in range(7):
        real[i] = line[i+60]

    output = net(x)

    loss = loss_func(output, real)
    x_axis.append(t)
    y_axis.append(loss.data.numpy())
    #print(loss.data.numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_np = np.array(x_axis)
y_np = np.array(y_axis)
plt.figure()
plt.plot(x_np, y_np, color='green', linewidth=0.5, linestyle='-')
plt.show()