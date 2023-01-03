import random
import time

import numpy as np
import torch

from torch import nn


def to_visit(coord, n):
    possible = []
    if coord[0] + 1 < n:
        possible.append((coord[0] + 1, coord[1]))
    if coord[0] - 1 >= 0:
        possible.append((coord[0] - 1, coord[1]))
    if coord[1] + 1 < n:
        possible.append((coord[0], coord[1] + 1))
    if coord[1] - 1 >= 0:
        possible.append((coord[0], coord[1] - 1))
    return possible


def solvable(grid, n):
    visited = set((0, 0))
    get_neigh = [(0, 0)]
    over = False
    while over == False:
        current = get_neigh.pop()
        possible = [x for x in to_visit(current, n) if grid[x] != 1 and x not in visited]
        visited.update(possible)
        get_neigh += possible
        if len(get_neigh) == 0:
            over = True

    if (n - 1, n - 1) in visited:
        return True
    else:
        return False


def generate_grid(n, wall_freq=0.5):
    is_solvable = False
    while is_solvable == False:
        grid = [0]
        for _ in range(n * n - 1):
            if np.random.rand() < wall_freq:
                grid.append(1)
            else:
                grid.append(0)
        grid[-1] = 0
        grid = np.array(grid).reshape(n, n)
        grid[-1][-1] = 7
        is_solvable = solvable(grid, n)
    return grid


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stack = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.input_size**2),
            nn.PReLU(),
            nn.Linear(self.input_size**2, self.output_size),
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
        training = True
        while training:
            games_played = 0
            inputs = []
            outputs = []
            while games_played < 10:
                status = self.game.status
                if status in ["WIN", "LOSE"]:
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
            self.game.reset()
            state = self.game.state
            while self.game.status not in ["WIN", "LOSE"]:
                pred = torch.argmax(self.model(torch.Tensor(state))).item()
                self.game.move(pred)
                state = self.game.state

            if self.game.status == "WIN":
                training = False

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
