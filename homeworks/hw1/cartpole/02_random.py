# Из учебного комплекта «Первая нейросеть в игровой среде», 02.11.2026.
# Происхождение и порядок запуска: ../README.md.
"""Пять эпизодов со случайными действиями; --headless без окна."""
import argparse
import gymnasium as gym

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
env = gym.make("CartPole-v1", render_mode=None if args.headless else "human")
env.action_space.seed(42)
try:
    for episode in range(5):
        observation, info = env.reset(seed=42 + episode)
        total = 0.0
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        print(f"episode={episode + 1} return={total:.0f} "
              f"terminated={terminated} truncated={truncated}")
finally:
    env.close()
