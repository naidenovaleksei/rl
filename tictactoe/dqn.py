import random
import math

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, exptuple):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exptuple
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DenseNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_size=256):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_size, layer_size)
        self.l2 = nn.Linear(layer_size, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x


class DenseDuelingNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_size=256):
        nn.Module.__init__(self)
        self.fc_common = nn.Linear(input_size, layer_size)
        self.fc_adv = nn.Linear(layer_size, output_size)
        self.fc_val = nn.Linear(layer_size, 1)
        self.num_actions = output_size

    def forward(self, x):
        x = F.relu(self.fc_common(x))
        adv = torch.tanh(self.fc_adv(x))
        val = torch.tanh(self.fc_val(x)).expand(x.size(0), self.num_actions)
        mean_adv = adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)

        x = val + adv - mean_adv
        return x


class DQN:
    def __init__(
        self, env, gamma=0.8, eps_decay=1000, model_kind="dense", device="cpu"
    ):
        self.env = env
        if model_kind == "dense":
            self.model = DenseNetwork(
                input_size=env.state_dim, output_size=env.action_dim
            )
        elif model_kind == "dueling_dense":
            self.model = DenseDuelingNetwork(
                input_size=env.state_dim, output_size=env.action_dim
            )
        self.model = self.model.to(device)
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.model.parameters(), 0.001)
        self.steps_done = 0
        self.episode_durations = []

        self.gamma = gamma
        self.batch_size = 64

        self.eps_init, self.eps_final, self.eps_decay = 0.9, 0.05, eps_decay
        self.num_step = 0
        self.device = device

    def select_greedy_action(self, state, empty_spaces):
        state = state.to(self.device)
        outputs = self.model(state).detach().cpu()
        idx = empty_spaces[:, 0] * self.env.n_cols + empty_spaces[:, 1]
        mask = np.repeat(-np.inf, outputs.shape[1])
        mask[idx] = 0
        mask = torch.tensor([mask], dtype=torch.float32)
        res = (outputs.data + mask).data.max(1)[1].view(1, 1)
        # print(outputs, mask, res)
        return res

        # outputs = self.model(state).data.numpy()[0]
        # outputs = outputs.reshape(self.env.n_cols, self.env.n_rows)
        # return np.argmax([outputs[i[0], i[1]] for i in empty_spaces])

    def select_action(self, state, empty_spaces):
        sample = random.random()
        self.num_step += 1
        eps_threshold = self.eps_final + (self.eps_init - self.eps_final) * math.exp(
            -1.0 * self.num_step / self.eps_decay
        )
        if sample > eps_threshold:
            return self.select_greedy_action(state, empty_spaces)
        else:
            idx = empty_spaces[:, 0] * self.env.n_cols + empty_spaces[:, 1]
            action = np.random.choice(idx)
            return torch.tensor([[action]], dtype=torch.int64)

    def run_episode(self, e=0, do_learning=True, greedy=False, render=False):
        (state, empty_spaces) = self.env.reset()
        num_step = 0
        while True:
            if render:
                self.env.render()

            state_tensor = torch.tensor([state], dtype=torch.float32)
            with torch.no_grad():
                if greedy:
                    action = self.select_greedy_action(state_tensor, empty_spaces)
                else:
                    action = self.select_action(state_tensor, empty_spaces)
            action_item = action.item()
            action_item = [
                action_item // self.env.n_cols,
                action_item % self.env.n_cols,
            ]
            (next_state, empty_spaces), reward, done, _ = self.env.step(action_item)
            next_state_tensor = torch.tensor([next_state], dtype=torch.float32)

            transition = (
                state_tensor,
                action,
                # torch.tensor([[action]], dtype=torch.int64),
                next_state_tensor,
                torch.tensor([reward], dtype=torch.float32),
            )
            self.memory.store(transition)

            if do_learning:
                self.learn()

            state = next_state
            num_step += 1

            if done:
                self.episode_durations.append(num_step)
                break
        return reward

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # берём мини-батч из памяти
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        batch_state = Variable(torch.cat(batch_state)).to(self.device)
        batch_action = Variable(torch.cat(batch_action)).to(self.device)
        batch_reward = Variable(torch.cat(batch_reward)).to(self.device)
        batch_next_state = Variable(torch.cat(batch_next_state)).to(self.device)

        # считаем значения функции Q
        Q = self.model(batch_state).gather(1, batch_action).reshape([self.batch_size])

        # оцениваем ожидаемые значения после этого действия
        Qmax = self.model(batch_next_state).detach().max(1)[0]
        Qnext = batch_reward + (self.gamma * Qmax)

        # и хотим, чтобы Q было похоже на Qnext -- это и есть суть Q-обучения
        loss = F.smooth_l1_loss(Q, Qnext)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, num_experiments):
        reward_list = []
        for e in tqdm(range(num_experiments)):
            reward = self.run_episode(e)
            reward_list.append(reward)
        avg_reward = np.cumsum(reward_list) / (np.arange(len(reward_list)) + 1)
        return avg_reward
