# Из учебного комплекта «Первая нейросеть в игровой среде», 02.11.2026.
# Происхождение и порядок запуска: ../README.md.
"""Обучение с учителем на CPU, без PPO/DQN и без запуска Hollow Knight."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from model import make_model

torch.manual_seed(42)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
OUT = Path(__file__).resolve().parents[1] / "output"
data = np.load(OUT / "demonstrations.npz")
x = torch.from_numpy(data["x"])
y = torch.from_numpy(data["y"])
train_mask = torch.from_numpy(data["episode"] < 48)
x_train, y_train = x[train_mask], y[train_mask]
x_val, y_val = x[~train_mask], y[~train_mask]
model = make_model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
history = []
print(model)
print("Parameters:", sum(p.numel() for p in model.parameters()))
for epoch in range(1, 31):
    model.train()
    order = torch.randperm(len(x_train))
    loss_sum = 0.0
    for indices in order.split(128):
        logits = model(x_train[indices])
        loss = criterion(logits, y_train[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(indices)
    model.eval()
    with torch.no_grad():
        accuracy = (model(x_val).argmax(1) == y_val).float().mean().item()
    row = {"epoch": epoch, "loss": loss_sum / len(x_train),
           "validation_accuracy": accuracy}
    history.append(row)
    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch={epoch:02d} loss={row['loss']:.4f} val_acc={accuracy:.3f}")
torch.save(model.state_dict(), OUT / "policy.pt")
(OUT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
print("Saved output/policy.pt")
