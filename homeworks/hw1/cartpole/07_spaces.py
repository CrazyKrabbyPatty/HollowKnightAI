# Из учебного комплекта «Первая нейросеть в игровой среде», 02.11.2026.
# Происхождение и порядок запуска: ../README.md.
"""Небольшие исполняемые опыты со Spaces и обёртками."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

actions = spaces.Discrete(2, seed=42)
print("Discrete:", actions.sample(), actions.contains(1), actions.contains(2))
coordinates = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32, seed=42)
print("Box:", coordinates.sample(), coordinates.contains(np.array([0, 0], dtype=np.float32)))
state = spaces.Dict({"position": coordinates, "mode": spaces.Discrete(3)})
state.seed(42)
print("Dict:", state.sample())
env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1", max_episode_steps=5))
try:
    obs, info = env.reset(seed=42)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            print("End:", terminated, truncated, info)
            break
finally:
    env.close()
