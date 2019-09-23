#!/usr/bin/env python3
from lab3.my_agents import MyAgent
from misio.lost_wumpus.testing import test_locally
from misio.lost_wumpus.agents import RandomAgent, SnakeAgent
import numpy as np

np.set_printoptions(precision=3, suppress=True)
n = 1

test_locally("tests/2015.in", MyAgent, n=n)
test_locally("tests/2015.in", SnakeAgent, n=n)
test_locally("tests/2015.in", RandomAgent, n=n)
