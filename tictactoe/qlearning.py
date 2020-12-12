import numpy as np
from tqdm import tqdm
from strategy import RandomStrategy


NUM_EXPERIMENTS_DEFAULT = 100000


def q_learning(
    env,
    pi,
    play_as_crosses,
    opponent=None,
    eps=0.05,
    alpha=0.05,
    gamma=0.05,
    num_experiments=NUM_EXPERIMENTS_DEFAULT,
):
    Q = pi.Q
    if opponent is None:
        opponent = RandomStrategy()
    reward_list = []
    for _ in tqdm(range(num_experiments)):
        observation = env.reset()
        done = False
        if not play_as_crosses:
            _, empty_spaces_op, _ = observation
            At_op = np.random.randint(len(empty_spaces_op))
            observation, _, done, _ = env.step(empty_spaces_op[At_op])
        while not done:
            # player's step
            _hash, empty_spaces, cur_turn = observation
            assert play_as_crosses == (cur_turn == 1)
            At = pi.getActionEpsGreedy(_hash, empty_spaces, eps)
            St = observation
            observation, reward, done, _ = env.step(empty_spaces[At])
            if not play_as_crosses:
                reward = -reward
            assert pi.check_state(St[0])
            if done:
                Q[St[0]][At] += alpha * (reward - Q[St[0]][At])
                break

            # opponent's step
            _hash_op, empty_spaces_op, _ = observation
            At_op = opponent.getActionGreedy(_hash_op, empty_spaces_op)
            observation, reward, done, _ = env.step(empty_spaces_op[At_op])
            if not play_as_crosses:
                reward = -reward
            if done:
                Q[St[0]][At] += alpha * (reward - Q[St[0]][At])
                break
            assert pi.check_state(observation[0])
            Q[St[0]][At] += alpha * (
                reward + gamma * Q[observation[0]].max() - Q[St[0]][At]
            )
        reward_list.append(reward)
    return np.cumsum(reward_list) / (np.arange(len(reward_list)) + 1)
