import torch
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) prepare data
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise= 20, random_state=1)

X = torch.from_numpy(x_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))
y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

# 1 ) Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) loss and optimizer
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # Backward pass
    loss.backward()

    # Update
    optimizer.step()

    optimizer.zero_grad()

    if(epoch+1)%10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.3f}")

# Plotting
predicted = model(X).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()

