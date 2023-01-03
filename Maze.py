import time

import streamlit as st
import streamlit.components.v1 as components

from src.games import Game
from src.utils import NeuralNetwork, Qlearning, generate_grid


st.title("Maze solver")

col1, col2 = st.columns(2)

with col1:
    n = st.number_input("Size of maze", min_value=3, max_value=8, value=3)

with col2:

    freq = st.slider("Frequency of walls", min_value=0.0, max_value=1.0, value=0.5)


with open("main.html", "r", encoding="utf-8") as f:
    text = f.read()

if st.button("Solve"):
    grid = generate_grid(n, freq)
    game = Game(grid=grid)
    model = NeuralNetwork(input_size=n**2, output_size=4)
    q = Qlearning(model=model, game=game)

    with st.empty():
        col1, col2 = st.columns(2)
        with col1:
            components.html(text + game.html(), height=300)
        with col2:
            with st.spinner("Training"):
                q.train()

        B = q.play()
        for x in B:
            components.html(text + x, height=500)
            time.sleep(0.3)
        components.html(text + B[-1], height=500)
