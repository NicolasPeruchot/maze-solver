from utils import Qlearning, NeuralNetwork
from games import Game
import torch
import numpy as np

game = Game(
    grid=[
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
    ]
)

model = NeuralNetwork(input_size=16, output_size=4)

q = Qlearning(model=model, game=game)

q.train()

q.play()
