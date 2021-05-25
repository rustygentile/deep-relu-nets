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

n = 1000
x = torch.linspace(-1, 1, n).reshape([-1,1])
y = torch.rand(x.shape)
x_fine = torch.linspace(-1, 1, n * 10).reshape([-1,1])

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

n_train_steps = 10001
print_interval = n_train_steps // 10
losses = np.empty(n_train_steps)
p_trn = {}

for step in range(n_train_steps):

    optimizer.zero_grad()
    loss = nn.MSELoss()(net(x), y)
    loss.backward()
    optimizer.step()

    losses[step] = loss
    if print_interval > 0 and step % print_interval == 0:
        print(f'step: {step}, error: {loss}')
        p_trn[str(step)] = net(x_fine).detach().numpy()


print(f'\nseed {seed}, manual {manual}')


if __name__ == '__main__':

    fig, ax = plt.subplots(1, 2)
    ax[0].set_yscale('log')
    ax[0].plot(losses, label='Losses')
    plt0, = ax[0].plot(step, losses[-1], '.', label=str(step))
    ax[0].legend()

    ax[1].plot(x.detach().numpy(), y.detach().numpy(), label='CPwL')
    plt1, = ax[1].plot(x_fine.detach().numpy(), p_trn[str(step)], '--', label='ReLU')
    ax[1].legend()
	
    slider_ax = plt.axes([0, 0, 1, 0.05])
    slider = plt.Slider(
        ax=slider_ax,
        label='Training Step',
        valmin=0,
        valmax=10,
        valinit=10
        )

    def update(val):
        step = int(val) * print_interval
        plt0.set_xdata(step)
        plt0.set_ydata(losses[step])
        plt0.set_label(str(step))
        plt1.set_ydata(p_trn[str(step)])

    update(10)
    slider.on_changed(update)
    plt.show()
