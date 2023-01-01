import numpy as np

from games import Game
from utils import NeuralNetwork, Qlearning, generate_grid


grid = generate_grid(4)

game = Game(grid=grid)

model = NeuralNetwork(input_size=len(grid) ** 2, output_size=4)

q = Qlearning(model=model, game=game)

q.train()
