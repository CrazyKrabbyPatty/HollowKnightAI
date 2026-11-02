# Полезные материалы для HollowKnightAI

Подборка на **02.11.2026**. Эти источники помогают выбрать основу среды, алгоритм обучения и способ наблюдения за игрой. Для первого эксперимента особенно полезны готовые интерфейсы игры, описания функций награды и ограничения воспроизводимости.

## Научные тексты

- **Jennifer Lee, 2023 — [Adapting Model-Free Reinforcement Learning for Boss Fights in Hollow Knight: a reward shaping approach](https://theses.liacs.nl/pdf/2022-2023-LeeJ.pdf).** Бакалаврская работа Leiden University полезна для проектирования награды: на Marmu показано, как дополнительные поощрения могут ускорять обучение или закреплять поведение без победы, а попытка переноса на Traitor Lord выявляет ограничения обобщения.
- **Anthony Maltsev, 2025 — [A Truly Hollow Knight: Resource Constrained Visually Dense Boss Fights](https://cs224r.stanford.edu/spring_2025/projects/pdfs/CS224R_Project_Report1.pdf).** Отчёт курса Stanford CS224R полезен как компактный пример сравнения DQN, его вариантов и Discrete SAC на Gruz Mother при обучении на CPU, с изображениями и заранее заданными действиями.
- **Weipu Zhang и соавторы, ICLR 2026 — [Object-Centric World Models from Few-Shot Annotations for Sample-Efficient Reinforcement Learning](https://arxiv.org/abs/2501.16443).** Конференционная статья об OC-STORM полезна для будущих экспериментов с моделью мира и объектными признаками из изображений; раннее название *Objects matter…* относится к той же работе, а вычислительные требования заметно выше, чем у простых базовых решений.

## Ключевые репозитории

- **Ruifeng Cui — [DQN_HollowKnight](https://github.com/ailec0623/DQN_HollowKnight).** Ранняя реализация DQN полезна как пример управления через Win32 и получения скриншотов и параметров игры, с конкретными зависимостями от Windows, CUDA и геометрии окна.
- **Zhantao Yang — [HollowKnight_RL](https://github.com/seermer/HollowKnight_RL).** Проект полезен для визуального DQN: содержит улучшения алгоритма, журнал обучения Hornet и выделенную работавшую ветку `hornet_new`; требует Windows, CUDA и модов здоровья.
- **Aditya Jain — [HKRL](https://github.com/AdityaJain1030/HKRL).** Реализации DQN/PPO и C#-мод полезны как пример Gym-подобного интерфейса через WebSocket, представления игровых данных и сбора опыта в нескольких экземплярах игры.
- **S. Sun — [rl-hollowknight-zote](https://github.com/simonsunyiming/rl-hollowknight-zote).** Переработка среды Yang с PPO из Stable-Baselines3 полезна для изучения частоты действий, управления высотой прыжка и боя с Zote; прежний адрес `Moapacha` относится к этому же проекту.

Результаты относятся преимущественно к отдельным боям с боссами; подтверждённого полного прохождения игры в подборке нет, а доли побед при разных боссах, наградах, сохранениях и бюджетах обучения нельзя сравнивать как единый тест.
