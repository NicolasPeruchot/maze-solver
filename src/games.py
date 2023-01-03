import numpy as np


class Game:
    def __init__(self, grid) -> None:
        self.n = len(grid)
        self.start = grid
        self.actions = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        self.rewards = {
            "INVALID": -0.2 * self.n,
            "WALL": -0.1 * self.n,
            "VALID": -0.05 * self.n,
            "VISITED": -0.1 * self.n,
            "WIN": (self.n) / 4,
            "LOSE": -0.1 * ((self.n) ** 2),
        }
        self.max_moves = (self.n) ** 3
        self.reset()

    def reset(self):
        self.n_moves = 0
        self.visited = set()
        self.position = (0, 0)
        self.status = "START"
        self.get_state()

    def get_state(self):
        self.grid = np.copy(self.start)
        self.grid[self.position] = 8
        self.state = self.grid.reshape(1, -1)

    def outbound(self, action):
        if self.position[0] == 0 and action == "UP":
            return True
        elif self.position[1] == 0 and action == "LEFT":
            return True
        elif self.position[0] == self.n - 1 and action == "DOWN":
            return True
        elif self.position[1] == self.n - 1 and action == "RIGHT":
            return True
        return False

    def html(self):
        colors = {1: "content_wall", 0: "content_empty", 8: "content_pos", 7: "content_exit"}
        text = "<table>"
        for line in range(self.n):
            current = "<tr>"
            for col in range(self.n):
                current += f"""
                <td>
                    <div class="{colors[self.grid[line][col]]}"> </div>
                </td>"""
            current += "</tr>"

            text += current
        text += "<table>"
        return text

    def move(self, action):
        self.n_moves += 1

        action = self.actions[action]

        if self.n_moves > self.max_moves:
            self.status = "LOSE"
            return (self.position, self.status, self.rewards[self.status])

        if self.outbound(action):
            self.status = "INVALID"
            return (self.position, self.status, self.rewards[self.status])

        else:

            next_position = self.position
            if action == "UP":
                next_position = (next_position[0] - 1, next_position[1])
            elif action == "DOWN":
                next_position = (next_position[0] + 1, next_position[1])
            elif action == "LEFT":
                next_position = (next_position[0], next_position[1] - 1)
            elif action == "RIGHT":
                next_position = (next_position[0], next_position[1] + 1)

            if self.start[next_position] == 1:
                self.status = "WALL"

            else:
                self.position = next_position
                if self.position == (self.n - 1, self.n - 1):
                    self.status = "WIN"

                elif self.position in self.visited:
                    self.status = "VISITED"

                else:
                    self.status = "VALID"

                self.visited.add(self.position)
            self.get_state()
            return (self.position, self.status, self.rewards[self.status])
