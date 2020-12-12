from collections import defaultdict
import numpy as np
from tqdm import tqdm

from game import get_move
from wrapper import BaseWrapper
from strategy import RandomStrategy
from copy import deepcopy


class MCTree:
    def __init__(self):
        self.stats_dict = dict()
        self.c = np.sqrt(2)

    def get_best_action(self, state, actions):
        t = self.stats_dict.get(state, (0, 0))[1]
        if t == 0:
            return None
        best_action = max(actions, key=lambda action: self.get_score(action, state, t))
        return best_action

    def get_score(self, action, state, t):
        key = (state, tuple(action))
        w, n = self.stats_dict.get(key, (0, 0))
        if n > 0:
            score = w / n + self.c * np.sqrt(np.log(t) / n)
        else:
            score = 0
        return score

    def update(self, key, w, n):
        val = self.stats_dict.get(key, (0, 0))
        self.stats_dict[key] = (val[0] + w, val[1] + n)


def rollout(env, pi1, pi2, action=None):
    env = deepcopy(env)
    if action is not None:
        env.step(action)
    reward = env.isTerminal()
    done = reward is not None
    while not done:
        state, actions = env.getHash(), env.getEmptySpaces()
        if env.curTurn == 1:
            a = get_move(env, pi1, state, actions, random=False)
        else:
            a = get_move(env, pi2, state, actions, random=False)
        observation, reward, done, info = env.step(actions[a])
    return reward


def check_action(env, pi1, pi2, action, n_iter=10):
    rewards = [rollout(env, pi1, pi2, action) for _ in range(n_iter)]
    rewards = np.array(rewards)
    return np.sum(rewards > 0), np.sum(rewards < 0), len(rewards)


def train_mcts(env, mctree, play_as_crosses, n_iters=1000):
    wrapper = BaseWrapper(
        env, play_as_crosses=play_as_crosses, opponent=RandomStrategy()
    )
    reward_list = []
    for _ in tqdm(range(n_iters)):
        # begin from root
        state, actions = wrapper.reset()
        # select next node (action)
        next_action = 0
        state_action_list = []
        done = False
        while not done:
            state_action_list.append(state)
            action = mctree.get_best_action(state, actions)
            # select until leaf node (best action = None)
            if action is None:
                break
            state_action_list.append((state, tuple(action)))
            (state, actions), reward, done, _ = wrapper.step(action)
        # select random action
        if done:
            w, n = ((reward == 1) == play_as_crosses), 1
            reward_list.append(2 * w - 1)
        else:
            # select random action
            index = np.random.randint(len(actions))
            action = actions[index]
            state_action_list.append((state, tuple(action)))
            assert wrapper.env.getHash() == state
            # check_action
            if play_as_crosses:
                w, _, n = check_action(
                    wrapper.env, RandomStrategy(), RandomStrategy(), action, n_iter=1
                )
            else:
                _, w, n = check_action(
                    wrapper.env, RandomStrategy(), RandomStrategy(), action, n_iter=1
                )
            reward_list.append(2 * w - 1)
        # update all nodes
        for key in state_action_list:
            mctree.update(key, w, n)
    return np.cumsum(reward_list) / (np.arange(len(reward_list)) + 1)
