from itertools import combinations
import numpy as np


NUM_EXPERIMENTS_DEFAULT = 100000


def check_strategy(
    env,
    pi1,
    pi2,
    random_crosses,
    random_naughts,
    num_experiments=NUM_EXPERIMENTS_DEFAULT,
):
    """Играем много эпизодов по стратегии pi в режиме exploitation и возвращаем список наград за каждую игру"""
    total_reward = []

    for _ in range(num_experiments):
        env.reset()
        reward = test_game(env, pi1, pi2, random_crosses, random_naughts)
        total_reward.append(reward)
    env.close()

    print("Доля выигрышей крестиков: %.3f" % np.mean(np.array(total_reward) > 0))
    print("Доля выигрышей ноликов: %.3f" % np.mean(np.array(total_reward) < 0))
    avg_reward = np.cumsum(total_reward) / np.arange(1, len(total_reward) + 1)
    return avg_reward


def get_move(env, pi, s, actions, random=False):
    if random:
        return np.random.randint(len(actions))
    else:
        return pi.getActionGreedy(s, actions)


def test_game(env, pi1, pi2, random_crosses=False, random_naughts=True):
    """Играем тестовую партию между стратегиями или со случайными ходами, рисуем ход игры"""
    done = False
    env.reset()
    while not done:
        s, actions = env.getHash(), env.getEmptySpaces()
        if env.curTurn == 1:
            a = get_move(env, pi1, s, actions, random=random_crosses)
        else:
            a = get_move(env, pi2, s, actions, random=random_naughts)
        observation, reward, done, info = env.step(actions[a])
    return reward


def get_empty_spaces(state, n_cols):
    return [(i // n_cols, i % n_cols) for i, a in enumerate(state) if a == "1"]


def get_hashes_map(n_rows, n_cols):
    n = n_rows * n_cols
    hashes_dict_x = set()
    hashes_dict_o = set()
    for filled_cells_num in range(n + 1):
        for comb in combinations(range(n), filled_cells_num):
            naughts_count = filled_cells_num // 2
            for naught_comb in combinations(comb, naughts_count):
                hash_arr = np.ones(n, dtype=np.int8)
                hash_arr[list(comb)] = 2
                hash_arr[list(naught_comb)] = 0
                hash_str = "".join(hash_arr.astype(str))
                if filled_cells_num % 2 == 0:
                    hashes_dict_x.add(hash_str)
                elif 1 in hash_arr:
                    hashes_dict_o.add(hash_str)
    hashes_map_x = {h: get_empty_spaces(h, n_cols) for h in hashes_dict_x}
    hashes_map_o = {h: get_empty_spaces(h, n_cols) for h in hashes_dict_o}
    return hashes_map_x, hashes_map_o
