# Из учебного комплекта «Первая нейросеть в игровой среде», 02.11.2026.
# Происхождение и порядок запуска: ../README.md.
"""Сравнение на одинаковых 20 новых начальных состояниях."""
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from model import make_model, choose_action, teacher

torch.manual_seed(42)
torch.set_num_threads(1)
OUT = Path(__file__).resolve().parents[1] / "output"
untrained = make_model().eval()
trained = make_model()
trained.load_state_dict(torch.load(OUT / "policy.pt", map_location="cpu", weights_only=True))
trained.eval()
results = {}
for name in ("random", "untrained", "teacher", "trained"):
    env = gym.make("CartPole-v1")
    env.action_space.seed(2026)
    returns = []
    try:
        for seed in range(1000, 1020):
            obs, info = env.reset(seed=seed)
            total = 0.0
            while True:
                if name == "random":
                    action = env.action_space.sample()
                elif name == "teacher":
                    action = teacher(obs)
                else:
                    action = choose_action(trained if name == "trained" else untrained, obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
    finally:
        env.close()
    results[name] = {"returns": returns, "mean": float(np.mean(returns)),
                     "min": min(returns), "max": max(returns)}
    print(name, f"mean={np.mean(returns):.1f} min={min(returns):.0f} max={max(returns):.0f}")
(OUT / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
