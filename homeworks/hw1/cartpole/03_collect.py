# Из учебного комплекта «Первая нейросеть в игровой среде», 02.11.2026.
# Происхождение и порядок запуска: ../README.md.
"""Собирает состояния и ответы правила; последние эпизоды идут в проверку."""
from pathlib import Path
import numpy as np
import gymnasium as gym
from model import teacher

OUT = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
env = gym.make("CartPole-v1")
observations, labels, episodes = [], [], []
try:
    for episode in range(60):
        obs, info = env.reset(seed=episode)
        while True:
            target = teacher(obs)
            observations.append(obs.copy())
            labels.append(target)
            episodes.append(episode)
            # Иногда ошибаемся намеренно, чтобы увидеть состояния вне
            # идеальной траектории. Метка всё равно берётся у правила.
            action = int(rng.integers(2)) if rng.random() < 0.2 else target
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
finally:
    env.close()
np.savez(OUT / "demonstrations.npz", x=np.asarray(observations, dtype=np.float32),
         y=np.asarray(labels, dtype=np.int64), episode=np.asarray(episodes))
print(f"Saved {len(labels)} pairs from 60 episodes")
print("train: episodes 0..47; validation: episodes 48..59")
