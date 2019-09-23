from unittest import TestCase
import numpy as np
from scipy.ndimage.filters import convolve

from build.lib.misio import Action
from lab3.my_agents import MyAgent


class TestMyAgent(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        map = np.eye(5, 5)
        map[3, 2] = 2
        self.my_agent = MyAgent(map, 0.1, 0.1, 0.1)

    def test__get_neighbours(self):
        tests = [(0, 1), (4, 4), (2, 2)]

        results = [[(0, 2), (0, 0), (1, 1), (4, 1)],
                   [(4, 0), (4, 3), (0, 4), (3, 4)],
                   [(2, 3), (2, 1), (3, 2), (1, 2)]]

        for test, result in zip(tests, results):
            assert self.my_agent._get_neighbours(test) == result

    def test__roll_histogram(self):
        self.my_agent.histogram = np.zeros((4, 4))
        self.my_agent.histogram[1, 0] = 1.
        rolled_histogram = self.my_agent._roll_histogram(Action.LEFT)
        convolved = convolve(rolled_histogram, self.my_agent.move_probability, mode='wrap')

        rolled_histogram1 = self.my_agent._roll_histogram(Action.RIGHT)
        convolved1 = convolve(rolled_histogram1, self.my_agent.move_probability, mode='wrap')

        rolled_histogram2 = self.my_agent._roll_histogram(Action.UP)
        convolved2 = convolve(rolled_histogram2, self.my_agent.move_probability, mode='wrap')

        rolled_histogram3 = self.my_agent._roll_histogram(Action.DOWN)
        convolved3 = convolve(rolled_histogram3, self.my_agent.move_probability, mode='wrap')

        self.fail()
