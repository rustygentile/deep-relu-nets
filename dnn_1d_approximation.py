import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class GD:

    def __init__(self, params, lr):

        self.params = list(params)
        self.lr = lr

    def step(self):

        for p in self.params:
            p.data.sub_(self.lr * p.grad.data)

    def zero_grad(self):

        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

seed = None
seed = 3262104886130042880

if seed == None:
    seed = torch.initial_seed()
    manual = False
else:
    torch.manual_seed(seed)
    manual = True

class Net(nn.Module):

    def __init__(self, width, depth):

        super().__init__()

        self.layer_first = nn.Linear(1, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.layer_last = nn.Linear(width, 1)

    def part(self, x, max_layer):

        x = F.relu(self.layer_first(x))

        for l in self.layers[:max_layer]:
            x = F.relu(l(x))

        return x

    def forward(self, x, max_layer=None):

        max_layer = len(self.layers)
        x = self.part(x, max_layer)
        x = self.layer_last(x)

        return x


net = Net(width=20, depth=5)

n = 50
x = torch.linspace(-1, 1, n).reshape([-1,1])
y = x**2
y = torch.sin(np.pi*10*x)
y = torch.rand(x.shape)

x_fine = torch.linspace(-1, 1, n * 10).reshape([-1,1])

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
# optimizer = GD(net.parameters(), lr=0.05)

n_train_steps = 70000
print_interval = n_train_steps // 10
losses = np.empty(n_train_steps)

for step in range(n_train_steps):

    optimizer.zero_grad()
    loss = nn.MSELoss()(net(x), y)
    loss.backward()
    optimizer.step()

    losses[step] = loss
    if print_interval > 0 and step % print_interval == 0:
        print(f'step: {step}, error: {loss}')
        fig, ax = plt.subplots()
        ax.plot(x.detach().numpy(), y.detach().numpy(), label='CPwL')
        ax.plot(x_fine.detach().numpy(), net(x_fine).detach().numpy(), '--', label=f'ReLU {step}')
        ax.legend()



print(f'\nseed {seed}, manual {manual}')


if __name__ == '__main__':

    fig, ax = plt.subplots(1, 3)
    ax[0].set_yscale('log')
    ax[0].plot(losses)

    ax[1].plot(x.detach().numpy(), y.detach().numpy(), label='CPwL')
    ax[1].plot(x_fine.detach().numpy(), net(x_fine).detach().numpy(), '--', label='ReLU')
    ax[1].legend()
	
    y = []
    depth = len(net.layers)
    for i in range(depth):
        y.append(net.part(x, i))
    width = y[0].shape[-1]

    slider_ax = plt.axes([0, 0, 1, 0.05])
    slider = plt.Slider(slider_ax, 'layer', 0, depth)

    plots = []
    for j in range(width):
        plots.append(ax[2].plot(x.detach().numpy(), y[i].detach().numpy()[:,j])[0])

    def update(val):
        i = np.abs(np.arange(depth) - val).argmin()
        for j in range(width):
            plots[j].set_ydata(y[i].detach().numpy()[:,j])
    update(0)

    slider.on_changed(update)

    plt.show()
