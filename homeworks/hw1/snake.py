# Источник: https://ratcatcher.ru/media/inf/ai_intro/kod.zip
# Учебный пример автора конспекта; алгоритм сохранён без изменений.
# Змейка, которая учится играть сама: игра, три нейрона и обучение с подкреплением на numpy.
# Запуск:  python snake.py            — обучить 300 партий и показать игру в консоли
#          python snake.py 1000       — обучить 1000 партий
import os
import sys
import time

import numpy as np

N = 12                                       # поле N×N клеток
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]    # вправо, вниз, влево, вверх — по часовой стрелке
rng = np.random.default_rng(0)


# ---------- 1. Игра ----------
def new_game():
    g = {"snake": [(6, 6), (5, 6), (4, 6)], "dir": 0, "hunger": 0}   # голова — первая клетка
    place_food(g)
    return g


def place_food(g):
    free = [(x, y) for x in range(N) for y in range(N) if (x, y) not in g["snake"]]
    g["food"] = free[rng.integers(len(free))]


def blocked(x, y, body):
    return not (0 <= x < N and 0 <= y < N) or (x, y) in body


def dist(g):
    (hx, hy), (fx, fy) = g["snake"][0], g["food"]
    return abs(fx - hx) + abs(fy - hy)


def step(g, action):                         # action: 0 — прямо, 1 — направо, 2 — налево
    g["dir"] = (g["dir"] + [0, 1, 3][action]) % 4
    d0 = dist(g)
    dx, dy = DIRS[g["dir"]]
    head = (g["snake"][0][0] + dx, g["snake"][0][1] + dy)
    ate = head == g["food"]
    body = g["snake"] if ate else g["snake"][:-1]                   # хвост уползает, если не ели
    if blocked(*head, body):
        return True, False, d0, d0                                  # died, ate, d0, d1
    g["snake"] = [head] + body
    g["hunger"] = 0 if ate else g["hunger"] + 1
    if ate:
        place_food(g)
    return False, ate, d0, (0 if ate else dist(g))


# ---------- 2. Что видит змейка: 8 чисел ----------
def see(g):
    hx, hy = g["snake"][0]
    body = g["snake"][:-1]
    d, r, l = DIRS[g["dir"]], DIRS[(g["dir"] + 1) % 4], DIRS[(g["dir"] + 3) % 4]
    fx, fy = g["food"][0] - hx, g["food"][1] - hy
    ahead, right = fx * d[0] + fy * d[1], fx * r[0] + fy * r[1]
    return np.array([blocked(hx + d[0], hy + d[1], body),          # опасно прямо
                     blocked(hx + r[0], hy + r[1], body),          # опасно справа
                     blocked(hx + l[0], hy + l[1], body),          # опасно слева
                     ahead > 0, right > 0, right < 0, ahead < 0,   # яблоко впереди / справа / слева / сзади
                     1], dtype=float)                              # всегда 1 (смещение)


# ---------- 3. Награда: её придумываем мы ----------
def reward(died, ate, d0, d1, hunger):
    if died:
        return -10
    if ate:
        return 10
    return 0


# ---------- 4. Мозг: три нейрона, по одному на действие ----------
W = np.zeros((3, 8))                         # строки: прямо, направо, налево
ALPHA, GAMMA, EPS = 0.01, 0.9, 0.1           # скорость обучения, важность будущего, доля случайных проб


def train(episodes):
    for ep in range(episodes):
        g, apples = new_game(), 0
        x = see(g)
        while True:
            q = W @ x                                               # выходы трёх нейронов
            a = rng.integers(3) if rng.random() < EPS else int(np.argmax(q))
            died, ate, d0, d1 = step(g, a)
            r = reward(died, ate, d0, d1, g["hunger"])
            apples += ate
            end = died or g["hunger"] >= 100                        # врезалась или 100 шагов без еды
            x2 = see(g) if not end else x
            target = r if end else r + GAMMA * np.max(W @ x2)       # награда + оценка будущего
            W[a] += ALPHA * (target - q[a]) * x                     # дельта-правило
            x = x2
            if end:
                break
        if (ep + 1) % 50 == 0:
            print(f"партия {ep + 1}: яблок {apples}")


# ---------- 5. Смотрим, как играет ----------
def show(g, text):
    cells = [["· "] * N for _ in range(N)]
    cells[g["food"][1]][g["food"][0]] = "()"
    for i, (x, y) in enumerate(g["snake"]):
        cells[y][x] = "██" if i else "▓▓"
    print("\033[H\033[J" + "\n".join("".join(row) for row in cells) + "\n" + text)


def watch(games=3, delay=0.08):
    os.system("")                            # включает ANSI-коды в консоли Windows
    for _ in range(games):
        g, apples, steps = new_game(), 0, 0
        while True:
            died, ate, *_ = step(g, int(np.argmax(W @ see(g))))
            apples, steps = apples + ate, steps + 1
            if died or g["hunger"] >= 100:
                break
            show(g, f"шаг {steps}, яблок {apples}")
            time.sleep(delay)
        print("разбилась" if died else "100 шагов без яблока", "— яблок:", apples)
        time.sleep(1.5)


if __name__ == "__main__":
    train(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
    np.set_printoptions(precision=1, suppress=True)
    print("Веса (строки: прямо, направо, налево; столбцы: опасно прямо, справа, слева,")
    print("яблоко впереди, справа, слева, сзади, смещение):")
    print(W)
    input("Enter — смотреть, как играет")
    watch()
