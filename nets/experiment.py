import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import uuid
import logging
from nets.net import ReLUNet

logger = logging.getLogger(__name__)


class TrainingExperiment:
    """
    Training experiment using a ReLU net for a 1D piecewise linear function.

    Methods
    -------
    run()
        Runs the experiment.
    plot()
        Visualize the results.
    """
    def __init__(self,
                 width=20,
                 depth=5,
                 alpha=0.05,
                 n_pts=10,
                 n_intervals=10,
                 interval_steps=1000,
                 seed=None,
                 data=None):
        """
        Parameters
        ----------
        width: int
            Width of the ReLU net
        depth: int
            Depth of the ReLU net
        alpha: float
            Learning rate
        n_pts: int
            Number of piecewise linear points
        n_intervals: int
            Number of training intervals
        interval_steps: int
            Training steps per interval
        seed: int, optional
            Seed for randomization
        data: dict, optional
            Training data. Must have keys 'x' and 'y'. If not provided,
            random data will be generated.
        """

        # Initial values
        self.interval_steps = interval_steps
        self.n_intervals = n_intervals
        self.n_train_steps = n_intervals * interval_steps + 1
        self.losses = np.empty(self.n_train_steps)
        self.uid = str(uuid.uuid4().hex)

        # Training data
        if data is None:
            self.x = torch.linspace(-1, 1, n_pts).reshape([-1, 1])
            self.y = torch.rand(self.x.shape)
        else:
            self.x = torch.from_numpy(data['x']).float().reshape([-1, 1])
            self.y = torch.from_numpy(data['y']).float().reshape([-1, 1])

        # Used for plotting and reporting results
        self.x_fine = torch.linspace(-1, 1, n_pts * 10).reshape([-1, 1])

        # Partially trained NN results
        self.p_trn = {}

        logger.info(f'Creating ReLU net with id: {self.uid} width: {width} depth: {depth}')
        self.net = ReLUNet(width=width, depth=depth)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=alpha)

        logger.info(f'Theoretical number of parameters: {self.count_manually()}')
        logger.info(f'Actual trainable parameters: {self.count_parameters()}')

        if seed is None:
            self.seed = torch.initial_seed()
            self.manual = False
        else:
            torch.manual_seed(seed)
            self.manual = True

    def run(self):
        """
        Runs a training experiment. Results data is stored in the experiment
        object for now.
        """
        for step in range(self.n_train_steps):
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.net(self.x), self.y)
            loss.backward()
            self.optimizer.step()
            self.losses[step] = loss

            if step % self.interval_steps == 0:
                # TODO: write results data to csv instead of storing in self
                logger.info(f'step: {step}, error: {loss}')
                self.p_trn[str(step)] = self.net(self.x_fine).detach().numpy()

    def plot(self, file_name=None, display=True, title=None):
        """
        Plots an animation of the training process.
        """
        fig, ax = plt.subplots(1, 2, figsize=(7, 4))
        step = self.n_train_steps - 1

        # Log plot of losses
        ax[0].set_yscale('log')
        ax[0].plot(self.losses, label='Loss')
        plt0, = ax[0].plot(step, self.losses[-1], '.', markersize=10,
                           label=f'Current Step')
        ax[0].legend(loc='upper right')

        # Plot the exact function and the ReLU
        ax[1].plot(self.x.detach().numpy(), self.y.detach().numpy(),
                   label='CPwL')
        plt1, = ax[1].plot(self.x_fine.detach().numpy(),
                           self.p_trn[str(step)], '--', label='ReLU')
        ax[1].legend(loc='upper right')

        def animate(val):
            current_step = int(val) * self.interval_steps
            plt0.set_xdata(current_step)
            plt0.set_ydata(self.losses[current_step])
            plt0.set_label(f'Step:\n{current_step}')
            plt1.set_ydata(self.p_trn[str(current_step)])
            return plt0, plt1

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=list(range(self.n_intervals + 1)) + [self.n_intervals] * 5)

        if title is not None:
            plt.suptitle(title)

        if file_name is not None:
            ani.save(file_name)

        if display:
            plt.show()

    def count_parameters(self):
        """
        Return
        ------
        Exact number of trainable parameters counted by Pytorch
        """
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def count_manually(self):
        """
        Return
        ------
        Theoretical number of parameters using equation (5)
        """
        L = self.net.depth
        W = self.net.width
        return W * (W + 1) * L - (W - 1) ** 2 + 2
