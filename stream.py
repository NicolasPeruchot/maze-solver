import time

import streamlit as st
import streamlit.components.v1 as components

from src.games import Game
from src.utils import NeuralNetwork, Qlearning, generate_grid


st.title("Maze solver")

n = st.slider("Size of maze", min_value=3, max_value=6, value=3)


with open("main.html", "r", encoding="utf-8") as f:
    text = f.read()

grid = generate_grid(n)


game = Game(grid=grid)


A = components.html(text + game.html(), height=300)


if st.button("Solve"):
    model = NeuralNetwork(input_size=n**2, output_size=4)
    q = Qlearning(model=model, game=game)
    q.train()
    B = q.play()
    A.empty()
    for x in B:
        current = components.html(text + x, height=300)
        time.sleep(0.5)
        current.empty()
    current = components.html(text + B[-1], height=300)
