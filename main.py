# Code 1: Y = 2 * X
import torch
import torch.nn as nn

X = torch.tensor([[1.0],[2.0],[3.0]])
Y = torch.tensor([[2.0],[4.0],[6.0]])

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(300):
    y_pred = model(X)
    loss = ((y_pred - Y)**2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(torch.tensor([[4.0]])))


# Code 2: Y = X + 5

import torch
import torch.nn as nn

X = torch.tensor([[1.0],[2.0],[3.0]])
Y = torch.tensor([[6.0],[7.0],[8.0]])

model = nn.Linear(1, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(300):
    y_pred = model(X)
    loss = ((y_pred - Y)**2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(torch.tensor([[4.0]])))


# Code 3: Learn simple line from small data

import torch
import torch.nn as nn

X = torch.tensor([[5.0],[10.0]])
Y = torch.tensor([[10.0],[20.0]])

model = nn.Linear(1, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(500):
    y_pred = model(X)
    loss = ((y_pred - Y)**2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(torch.tensor([[7.0]])))


