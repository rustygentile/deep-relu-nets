import unittest
from test.functions import fourier
from nets.experiment import TrainingExperiment
import numpy as np
import logging


class MultiLayerTest(unittest.TestCase):

    def setUp(self):
        """
        Generate training data for a random Fourier series.
        """

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

        n_terms = 5
        np.random.seed(seed=123456)
        a = np.random.rand(n_terms)
        b = np.random.rand(n_terms)
        x = np.linspace(0, 1, 10000)
        y = fourier(0, a, b, 0.2, x)
        self.data = {
            'x': x,
            'y': y
        }

    def test_depths_fourier(self):

        for d in range(1, 10):
            my_exp = TrainingExperiment(width=20, depth=d, data=self.data,
                                        n_intervals=100, seed=67890)
            my_exp.run()
            my_exp.plot(display=False,
                        file_name=f'./images/fourier_terms_{5}_width_{20}_depth_{d}.gif',
                        title=f'Terms: {5}, Width: {20}, Depth: {d}',
                        func_name='Fourier')


if __name__ == '__main__':
    unittest.main()
