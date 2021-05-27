from nets.experiment import TrainingExperiment
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def constant_points_varying_depth(n, n_intervals=10):

    np.random.seed(seed=12345)
    data = {
        'x': np.linspace(-1, 1, n),
        'y': np.random.rand(n)
    }

    for d in range(10):

        my_exp = TrainingExperiment(width=20, depth=d, n_pts=n, data=data,
                                    n_intervals=n_intervals, seed=67890)
        my_exp.run()
        my_exp.plot(display=False,
                    file_name=f'./images/pts_{n}_width_20_depth_{d}.gif',
                    title=f'Points: {n}, Width: 20, Depth: {d}')


def count_params_10x10():
    theo = []
    act = []
    diff = []
    ds = []
    ws = []
    for d in range(1, 10):
        for w in range(1, 10):
            my_exp = TrainingExperiment(width=w, depth=d)
            logger.info(f'N comparable parameters: {w ** 2 * d}')
            theo.append(my_exp.count_manually())
            act.append(my_exp.count_parameters())
            diff.append(my_exp.count_parameters() - my_exp.count_manually())
            ds.append(d)
            ws.append(w)

    plt.plot(theo, act, '.')
    plt.xlabel('Theoretical Number of Parameters')
    plt.ylabel('Actual Trainable Parameters')
    plt.show()

    w_rng = np.arange(10)
    diff_calc = w_rng * (w_rng + 1)
    plt.plot(w_rng, diff_calc)
    plt.plot(ws, diff, '.')
    plt.xlabel('Width')
    plt.ylabel('Diff = (Actual - Theoretical)')
    plt.show()


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    #constant_points_varying_depth(10, n_intervals=10)
    #constant_points_varying_depth(50, n_intervals=100)
    count_params_10x10()
