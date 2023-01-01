import numpy as np

from games import Game
from utils import NeuralNetwork, Qlearning


grid = [
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    0,
]

game = Game(grid=grid)

model = NeuralNetwork(input_size=len(grid), output_size=4)

q = Qlearning(model=model, game=game)

q.train()

q.play()
