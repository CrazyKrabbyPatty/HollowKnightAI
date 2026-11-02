# Из учебного комплекта «Первая нейросеть в игровой среде», 02.11.2026.
# Происхождение и порядок запуска: ../README.md.
"""Показ обученной сети в настоящем окне CartPole."""
from pathlib import Path
import gymnasium as gym
import torch
from model import make_model, choose_action

OUT = Path(__file__).resolve().parents[1] / "output"
model = make_model()
model.load_state_dict(torch.load(OUT / "policy.pt", map_location="cpu", weights_only=True))
model.eval()
env = gym.make("CartPole-v1", render_mode="human")
try:
    for seed in range(1000, 1003):
        obs, info = env.reset(seed=seed)
        while True:
            obs, reward, terminated, truncated, info = env.step(choose_action(model, obs))
            if terminated or truncated:
                break
finally:
    env.close()
