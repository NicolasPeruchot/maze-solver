import os
import random
import time

import numpy as np
import torch

from torch import nn
from tqdm import tqdm


def generate_grid(n):
    grid = [0]

    for _ in range(n * n - 1):
        if np.random.rand() < 0.2:
            grid.append(1)
        else:
            grid.append(0)
    grid[-1] = 0
    grid = np.array(grid).reshape(n, n)
    grid[-1][-1] = 0
    return grid


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stack = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.input_size),
            nn.PReLU(),
            nn.Linear(self.input_size, self.output_size),
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits


class Qlearning:
    def __init__(self, model: NeuralNetwork, game) -> None:
        self.exploration_proba = 0.1
        self.gamma = 0.99
        self.model = model
        self.game = game

    def train(self):
        self.game.reset()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in tqdm(range(1000)):
            games_played = 0
            games_won = 0
            inputs = []
            outputs = []
            while games_played < 10:
                status = self.game.status
                if status in ["WIN", "LOSE"]:
                    if status == "WIN":
                        games_won += 1
                    self.game.reset()
                    games_played += 1
                game_state = self.game.state
                inputs.append(game_state)

                y = self.model(torch.Tensor(game_state))

                if np.random.rand() < self.exploration_proba:
                    action = random.choice(list(self.game.actions.keys()))
                else:
                    action = torch.argmax(y).item()

                _, status, reward = self.game.move(action)

                next_state = self.game.state

                Q_sa = torch.max(self.model(torch.Tensor(next_state))).item()

                y = y.detach().numpy()[0]

                y[action] = reward + self.gamma * Q_sa

                outputs.append(y)
            inputs = torch.Tensor(np.array(inputs))
            outputs = torch.Tensor(np.array(outputs)).reshape(
                inputs.shape[0], 1, self.model.output_size
            )

            pred = self.model(inputs)
            loss = self.loss(pred, outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 99:
                print("Epoch: ", epoch, "Games played: ", games_played, "Games won: ", games_won)

    def play(self):
        self.game.reset()
        all = []
        state = self.game.state
        while self.game.status != "WIN":
            all.append(self.game.html())
            pred = torch.argmax(self.model(torch.Tensor(state))).item()
            self.game.move(pred)
            time.sleep(0.5)
            state = self.game.state
        all.append(self.game.html())
        return all
