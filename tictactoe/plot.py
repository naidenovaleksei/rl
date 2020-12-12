import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np


def plot_board(env, pi, showtext=True, verbose=True, fontq=20, fontx=60):
    """Рисуем доску с оценками из стратегии pi"""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    X, Y = np.meshgrid(np.arange(0, env.n_rows), np.arange(0, env.n_rows))
    Z = np.zeros((env.n_rows, env.n_cols)) + 0.01
    s, actions = env.getHash(), env.getEmptySpaces()
    if pi is not None and s in pi.Q:
        for i, a in enumerate(actions):
            Z[a[0], a[1]] = pi.Q[s][i]
    ax.set_xticks([])
    ax.set_yticks([])
    surf = ax.imshow(Z, cmap=plt.get_cmap("Accent", 10), vmin=-1, vmax=1)
    if showtext:
        for i, a in enumerate(actions):
            if pi is not None and s in pi.Q:
                ax.text(
                    a[1],
                    a[0],
                    "%.3f" % pi.Q[s][i],
                    fontsize=fontq,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="w",
                )
    for i in range(env.n_rows):
        for j in range(env.n_cols):
            if env.board[i, j] == -1:
                ax.text(
                    j,
                    i,
                    "O",
                    fontsize=fontx,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="w",
                )
            if env.board[i, j] == 1:
                ax.text(
                    j,
                    i,
                    "X",
                    fontsize=fontx,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="w",
                )
    cbar = plt.colorbar(surf, ticks=[0, 1])
    ax.grid(False)
    plt.show()


def plot_reward(reward, title, label=None):
    if label is None:
        plt.plot(reward)
    else:
        plt.plot(reward, label=label)
        plt.legend()
    plt.ylim([-1, 1])
    plt.title(title)
    _ = plt.grid()


def get_and_print_move(
    env, pi, s, actions, random=False, verbose=True, fontq=20, fontx=60
):
    """Делаем ход, рисуем доску"""
    plot_board(env, pi, fontq=fontq, fontx=fontx)
    if verbose and (pi is not None):
        if s in pi.Q:
            for i, a in enumerate(actions):
                print(i, a, pi.Q[s][i])
        else:
            print("Стратегия не знает, что делать...")
    if random:
        return np.random.randint(len(actions))
    else:
        return pi.getActionGreedy(s, actions)


def plot_test_game(
    env,
    pi1,
    pi2,
    random_crosses=False,
    random_naughts=True,
    verbose=True,
    fontq=20,
    fontx=60,
):
    """Играем тестовую партию между стратегиями или со случайными ходами, рисуем ход игры"""
    done = False
    env.reset()
    while not done:
        s, actions = env.getHash(), env.getEmptySpaces()
        if env.curTurn == 1:
            a = get_and_print_move(
                env,
                pi1,
                s,
                actions,
                random=random_crosses,
                verbose=verbose,
                fontq=fontq,
                fontx=fontx,
            )
        else:
            a = get_and_print_move(
                env,
                pi2,
                s,
                actions,
                random=random_naughts,
                verbose=verbose,
                fontq=fontq,
                fontx=fontx,
            )
        observation, reward, done, info = env.step(actions[a])
        if reward == 1:
            print("Крестики выиграли!")
            plot_board(env, None, showtext=False, fontq=fontq, fontx=fontx)
        if reward == -1:
            print("Нолики выиграли!")
            plot_board(env, None, showtext=False, fontq=fontq, fontx=fontx)
