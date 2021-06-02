import unittest
from test.functions import piecewise_linear
from nets.experiment import TrainingExperiment
import numpy as np
import logging


class SingleLayerTest(unittest.TestCase):

    def setUp(self):
        """
        Generate training data for a piecewise linear function.
        """

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

        n_knots = 50
        np.random.seed(seed=123456)
        knot_xs = np.linspace(-1, 1, n_knots)
        knot_ys = np.random.rand(n_knots)
        x, y = piecewise_linear(knot_xs, knot_ys, 10)
        self.data = {
            'x': x,
            'y': y
        }

    def test_example(self):

        w = 50
        d = 2

        my_exp = TrainingExperiment(width=w, depth=d, data=self.data,
                                    n_intervals=100, seed=67890)
        my_exp.run()
        my_exp.plot(display=False,
                    file_name=f'./example.gif',
                    title=f'Points: {50}, Width: {w}, Depth: {d}',
                    func_name='CPwL')


if __name__ == '__main__':
    unittest.main()
