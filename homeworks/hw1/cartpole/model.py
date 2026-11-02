# Из учебного комплекта «Первая нейросеть в игровой среде», 02.11.2026.
# Происхождение и порядок запуска: ../README.md.
"""Одна небольшая полносвязная сеть и правило, которое она изучает."""
import torch
from torch import nn

def make_model():
    return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

def teacher(observation):
    # Положительный угол означает наклон вправо.
    # Угловая скорость помогает начать движение заранее.
    _, _, angle, angular_velocity = observation
    return int(angle + 0.5 * angular_velocity > 0)

def choose_action(model, observation):
    x = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        return int(logits.argmax(dim=1).item())
