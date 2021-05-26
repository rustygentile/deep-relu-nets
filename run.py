from nets.experiment import TrainingExperiment
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    my_exp = TrainingExperiment()
    my_exp.run()
    my_exp.plot()
