import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class BaseWrapper:
    def __init__(self, env, play_as_crosses, opponent):
        self.env = env
        self.opponent = opponent
        self.play_as_crosses = play_as_crosses

    def step(self, action):
        (next_state, empty_spaces, cur_turn), reward, done, info = self.env.step(action)
        assert self.play_as_crosses == (cur_turn != 1)

        if not done:
            At_op = self.opponent.getActionGreedy(next_state, empty_spaces)
            action = empty_spaces[At_op]
            (next_state, empty_spaces, cur_turn), reward, done, info = self.env.step(
                action
            )

        if not self.play_as_crosses:
            reward = -reward
        next_state = self.modify_state(next_state)
        return (next_state, empty_spaces), reward, done, info

    def reset(self):
        state, empty_spaces, cur_turn = self.env.reset()
        if not self.play_as_crosses:
            At_op = self.opponent.getActionGreedy(state, empty_spaces)
            (state, empty_spaces, cur_turn), reward, done, _ = self.env.step(
                empty_spaces[At_op]
            )
        state = self.modify_state(state)
        return (state, empty_spaces)

    def modify_state(self, state):
        return state


class DQNWrapper(BaseWrapper):
    def __init__(self, env, play_as_crosses, opponent):
        super().__init__(env, play_as_crosses, opponent)
        self.state_dim = env.n_cols * env.n_rows
        self.action_dim = env.n_cols * env.n_rows
        self.n_cols, self.n_rows = env.n_cols, env.n_rows

    def modify_state(self, state):
        return [int(c) - 1 for c in state]
