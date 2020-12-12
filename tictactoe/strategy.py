import numpy as np
from game import get_empty_spaces


class BaseQStrategy:
    def __init__(self, hashes_map={}):
        self.Q = {
            state: np.random.normal(size=len(empty_spaces))
            for state, empty_spaces in hashes_map.items()
        }

    def getActionGreedy(self, state, actions):
        raise NotImplementedError


class RandomStrategy(BaseQStrategy):
    def getActionGreedy(self, state, actions):
        return np.random.randint(len(actions))


class GreedyStrategy(BaseQStrategy):
    def getActionGreedy(self, state, actions):
        a_star = self.Q[state].argmax()
        return a_star

    def check_state(self, state):
        return state in self.Q

    def getActionEpsGreedy(self, state, actions, eps):
        assert self.check_state(state)
        action_space_n = len(actions)
        assert len(self.Q[state]) == action_space_n
        a_star = self.Q[state].argmax()
        ps = np.ones(action_space_n) * eps / action_space_n
        ps[a_star] = 1 - eps + eps / action_space_n
        At = np.random.choice(np.arange(action_space_n), p=ps / ps.sum())
        return At


class LazyGreedyStrategy(GreedyStrategy):
    def __init__(self, n_cols, hashes_map={}):
        super(LazyGreedyStrategy, self).__init__()
        self.n_cols = n_cols

    def getActionGreedy(self, state, actions):
        action_space_n = len(actions)
        if state not in self.Q:
            return np.random.randint(action_space_n)
        assert len(self.Q[state]) == action_space_n
        a_star = self.Q[state].argmax()
        return a_star

    def check_state(self, state):
        if state not in self.Q:
            empty_spaces = get_empty_spaces(state, self.n_cols)
            actions = np.random.normal(size=len(empty_spaces))
            self.Q[state] = actions
        return True


class MCTSStrategy:
    def __init__(self, mctree):
        self.mctree = mctree

    def getActionGreedy(self, state, actions):
        action = self.mctree.get_best_action(state, actions)
        if action is None:
            return np.random.randint(len(actions))
        index = np.where((actions == action).min(1))[0][0]
        return index


class DQNStrategy:
    def __init__(self, dqn, n_cols):
        import torch

        self.dqn = dqn
        self.n_cols = n_cols

    def getActionGreedy(self, state, actions):
        import torch

        state = self.modify_state(state)
        state_tensor = torch.tensor([state], dtype=torch.float32).to(self.dqn.device)

        action = self.dqn.select_greedy_action(state_tensor, actions)
        action_item = action.item()
        actions = np.array(actions)
        idx = actions[:, 0] * self.n_cols + actions[:, 1]
        a_star = np.argwhere(idx == action_item)[0][0]
        return a_star

    def check_state(self, state):
        return True

    def modify_state(self, state):
        return [int(c) - 1 for c in state]
