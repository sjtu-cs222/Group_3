import torch
import Model
import DataPreprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas
import Encryption


learningRate = 0.05
epoch = 10000

netA = Model.BasicNetwork(60,100,7)
netB = Model.BasicNetwork(60,100,7)

optimizerA = torch.optim.SGD(netA.parameters(), learningRate)
optimizerB = torch.optim.SGD(netB.parameters(), learningRate)
loss_func = torch.nn.MSELoss()

csv_data, ave1 = DataPreprocess.open_1()

# show
x_axis = []
y_axis_A = []
y_axis_B = []

for t in range(epoch):
    xA = torch.Tensor(60).zero_()
    xB = torch.Tensor(60).zero_()
    real = torch.Tensor(7).zero_()

    line = DataPreprocess.read_1(csv_data, ave1, t)

    # distrbute information for 2 parties
    # private data got
    for i in range(30):
        xA[i] = line[i]
    for i in range(30):
        xB[i] = line[i+30]

    # encrypt the input vector from B and send to A
    # A decrypt into text and join it into its private input vector
    cypher_B2A = Encryption.encrypt(xB.data.numpy())
    text_A = Encryption.decrypt(cypher_B2A)
    for i in range(30):
        xA[i+30] = text_A[i]

    # encrypt the input vector from A and send to B
    # B decrypt into text and join it into its private input vector
    cypher_A2B = Encryption.encrypt(xA.data.numpy())
    text_B = Encryption.decrypt(cypher_A2B)
    for i in range(30):
        xB[i+30] = text_B[i]
        
    for i in range(7):
        real[i] = line[i+60]
    
    outputA = netA(xA)
    outputB = netB(xB)

    lossA = loss_func(outputA, real)
    lossB = loss_func(outputB, real)
    #print(lossA.data.numpy(),lossB.data.numpy())

    x_axis.append(t)
    y_axis_A.append(lossA.data.numpy())
    y_axis_B.append(lossB.data.numpy())

    optimizerA.zero_grad()
    lossA.backward()
    optimizerA.step()

    optimizerB.zero_grad()
    lossB.backward()
    optimizerB.step()

x_np = np.array(x_axis)
y_np_A = np.array(y_axis_A)
y_np_B = np.array(y_axis_B)
plt.figure()
plt.xlabel('epoches')
plt.ylabel('loss')
plt.plot(x_np, y_np_A, color='blue', linewidth=0.5, linestyle='-')
plt.plot(x_np, y_np_B, color='red', linewidth=0.5, linestyle='-')
plt.show()