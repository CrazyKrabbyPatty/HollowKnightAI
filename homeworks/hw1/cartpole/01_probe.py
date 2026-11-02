# Из учебного комплекта «Первая нейросеть в игровой среде», 02.11.2026.
# Происхождение и порядок запуска: ../README.md.
"""Проверка интерпретатора, пакетов и первого шага Gymnasium."""
import sys
from importlib.metadata import version
import gymnasium as gym

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
for package in ("gymnasium", "numpy", "torch", "pygame-ce"):
    print(package, version(package))
env = gym.make("CartPole-v1")
try:
    observation, info = env.reset(seed=42)
    env.action_space.seed(42)
    print("Observations:", env.observation_space)
    print("Actions:", env.action_space)
    print("reset:", observation, info)
    action = env.action_space.sample()
    print("action:", action)
    print("step:", env.step(action))
finally:
    env.close()
