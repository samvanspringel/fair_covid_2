import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import gym


class RewardSlicing(gym.Wrapper):
    def __init__(self, env, reward_indices):
        super().__init__(env)
        self.reward_indices = reward_indices

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # slice out only the desired indices, e.g. [1,2,3,4]
        # for (ARH, SB_W, SB_S, SB_L).
        reward = reward[self.reward_indices]
        return obs, reward, done, info


class MultiDiscreteAction(gym.ActionWrapper):
    def __init__(self, env, action_map):
        super(MultiDiscreteAction, self).__init__(env)
        self.action_map = action_map
        self.action_space = gym.spaces.MultiDiscrete([len(am) for am in action_map])

    def action(self, action):
        return np.array([self.action_map[i][action[i]] for i in range(len(self.action_map))])


class ScaleRewardEnv(gym.RewardWrapper):
    def __init__(self, env, min_=0., scale=1.):
        gym.RewardWrapper.__init__(self, env)
        self.min = min_
        self.scale = scale

    def reward(self, reward):
        return (reward - self.min) / self.scale


class TodayWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TodayWrapper, self).__init__(env)

    def reset(self):
        s = super(TodayWrapper, self).reset()
        if len(s) == 4:
            sb, ss, se, sa = s
            return (sb, ss[-1].T, se[-1], sa)
        else:
            ss, se, sa = s
            return (ss[-1].T, se[-1], sa)

    # step function of covid env returns simulation results of every day of timestep
    # only keep current day
    # also discard first reward
    def step(self, action):
        s, r, d, i = super(TodayWrapper, self).step(action)
        # sum all the social burden objectives together:
        p_tot = r[2:].sum()
        r = np.concatenate((r, p_tot[None]))
        if len(s) == 4:
            sb, ss, se, sa = s
            s = (sb, ss[-1].T, se[-1], sa)
        else:
            ss, se, sa = s
            s = (ss[-1].T, se[-1], sa)
        return s, r, d, i


class HistoryEnv(gym.Wrapper):
    def __init__(self, env, size=4):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.size = size
        # will be set in _convert
        self._state = None

        # history stacks observations on dim 0
        low = np.repeat(self.observation_space.low, self.size, axis=0)
        high = np.repeat(self.observation_space.high, self.size, axis=0)
        self.observation_space = gym.spaces.Box(low, high)

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        state = self.env.reset(**kwargs)
        # add history dimension
        s = np.expand_dims(state, 0)
        # fill history with current state
        self._state = np.repeat(s, self.size, axis=0)
        return np.concatenate(self._state, axis=0)

    def step(self, ac):
        state, r, d, i = self.env.step(ac)
        # shift history
        self._state = np.roll(self._state, -1, axis=0)
        # add state to history
        self._state[-1] = state
        return np.concatenate(self._state, axis=0), r, d, i


ss_emb = {
    'conv1d': nn.Sequential(
        nn.Conv1d(10, 20, kernel_size=3, stride=2, groups=5),
        nn.ReLU(),
        nn.Conv1d(20, 20, kernel_size=2, stride=1, groups=10),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(100, 64),
        nn.SiLU()
    ),
    'small': nn.Sequential(
        nn.Flatten(),
        nn.Linear(130, 64),
        nn.Sigmoid()
    ),
    'big': nn.Sequential(
        nn.Flatten(),
        nn.Linear(130, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.Sigmoid()
    ),
}

se_emb = {
    'small': nn.Sequential(
        nn.Linear(1, 64),
        nn.Sigmoid()
    ),
    'big': nn.Sequential(
        nn.Linear(1, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.Sigmoid()
    )
}

sa_emb = {
    'small': nn.Sequential(
        nn.Linear(3, 64),
        nn.Sigmoid()
    ),
    'big': nn.Sequential(
        nn.Linear(3, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.Sigmoid()
    )
}


class CovidModel(nn.Module):

    def __init__(self,
                 nA,
                 scaling_factor,
                 objectives,
                 ss_emb,
                 se_emb,
                 sa_emb,
                 with_budget=False):
        super(CovidModel, self).__init__()
        self.scaling_factor = scaling_factor[:, objectives + (len(scaling_factor) - 1,)]
        self.objectives = objectives
        self.ss_emb = ss_emb
        self.se_emb = se_emb
        self.sa_emb = sa_emb
        if with_budget:
            # sa_emb has 3 inputs (1 per action), same as budget (1 budget per action)
            self.sb_emb = copy.deepcopy(sa_emb)
        else:
            # otherwise, we include the number of steps left (single int),
            # se_emb takes a single input as well
            self.sb_emb = copy.deepcopy(se_emb)
        self.s_emb = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU()
        )
        self.c_emb = nn.Sequential(nn.Linear(self.scaling_factor.shape[-1], 64),
                                   nn.SiLU())
        self.fc = nn.Sequential(nn.Linear(64, 64),
                                nn.SiLU(),
                                nn.Linear(64, nA))

    def forward(self, state, desired_return, desired_horizon):
        # filter desired_return to only keep used objectives
        desired_return = desired_return[:, self.objectives]
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c * self.scaling_factor
        # if self.sb_emb is not None:
        sb, ss, se, sa = state
        s = self.ss_emb(ss.float()) * self.se_emb(se.float()) * self.sa_emb(sa.float()) * self.sb_emb(sb.float())
        # else:
        #     ss, se, sa = state
        #     s = self.ss_emb(ss.float())*self.se_emb(se.float())*self.sa_emb(sa.float())
        # concatenate embeddings
        # s = torch.cat((self.ss_emb(ss.float()), self.se_emb(se.float()), self.sa_emb(sa.float())), 1)
        # hadamard product on embeddings
        s = self.s_emb(s)
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        sc = s * c
        # sc = torch.cat((s, c), 1)
        log_prob = self.fc(sc)
        return log_prob


class DiscreteHead(nn.Module):
    def __init__(self, base):
        super(DiscreteHead, self).__init__()
        self.base = base

    def forward(self, state, desired_return, desired_horizon):
        x = self.base(state, desired_return, desired_horizon)
        x = F.log_softmax(x, 1)
        return x


class MultiDiscreteHead(nn.Module):
    def __init__(self, base):
        super(MultiDiscreteHead, self).__init__()
        self.base = base

    def forward(self, state, desired_return, desired_horizon):
        x = self.base(state, desired_return, desired_horizon)
        b, o = x.shape
        # hardcoded
        x = x.reshape(b, 3, 3)
        x = F.log_softmax(x, 2)
        return x


class ContinuousHead(nn.Module):
    def __init__(self, base):
        super(ContinuousHead, self).__init__()
        self.base = base

    def forward(self, state, desired_return, desired_horizon):
        x = self.base(state, desired_return, desired_horizon)
        x = torch.sigmoid(x)
        # bound in [0, 1]
        # x = (x+1)/2
        return x


def multidiscrete_env(env):
    # the discrete actions:
    a = [[0.0, 0.3, 0.6], [0, 0.5, 1], [0.3, 0.6, 0.9]]
    env = MultiDiscreteAction(env, a)
    return env
