import torch
import Model
import DataPreprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas


learningRate = 0.1
epoch = 2500

net = Model.BasicNetwork(5,6,1)

optimizer = torch.optim.SGD(net.parameters(), learningRate)
loss_func = torch.nn.MSELoss()

csv_data, ave0 = DataPreprocess.open_0()

# show
x_axis = []
y_axis = []

for t in range(epoch):
    x = torch.Tensor(5).zero_()
    real = torch.Tensor(1).zero_()

    line = DataPreprocess.read_0(csv_data, ave0, t)
    #print(line)

    for i in range(5):
        x[i] = line[i]
    real[0] = line[-1]

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