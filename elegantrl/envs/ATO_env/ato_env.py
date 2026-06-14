import random

import gym
import numpy as np
from gym import spaces
from elegantrl.envs.ATO_env.ato_config import ATOTestConfig, ATOTrainConfig
from scipy.stats import gamma


class ATOFixedLTCATest(gym.Env):
    def __init__(self,conf):
        self.config = conf
        self.env_name = 'ATOCA-v0'
        self.state_dim = len(self.config.observation_low)  # feature number of state
        self.action_dim = len(self.config.action_low)  # feature number of action
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.max_step = 10000  # the max step of each episode
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = 1e15

        self.IO = np.zeros(self.config.NoComponents)  # current inventory on-hand
        self.BO = np.zeros(self.config.NoProducts)  # current backorder
        self.IT = np.zeros([self.config.NoComponents, max(self.config.L)])  # inventory in transit
        self.D = np.zeros(self.config.NoProducts)
        self.curTime = 0  # current time
        self.action_space = spaces.Box(
            high=self.config.action_high,
            low=self.config.action_low
        )

        self.observation_space = spaces.Box(
            high=self.config.observation_high,
            low=self.config.observation_low
        )

    def step(self, action):
        action = [(action[i] + 1) * self.config.action_high[i] / 2 for i in
                  range(self.config.NoProducts + self.config.NoComponents)]
        assert self.action_space.contains(action)
        self._apply_action(action)

        reward = -(sum(self.config.c_p[i] * (self.BO[i] - self.D[i]) for i in range(self.config.NoProducts))
                   + sum(self.config.c_h[i] * self.IO[i] for i in range(self.config.NoComponents)))

        self.curTime += 1
        if self.curTime == self.config.T:
            done = True
        else:
            done = False

        info = {}
        return self._get_observation(), float(reward), done, info

    def reset(self):
        self.IO = np.zeros(self.config.NoComponents)  # current inventory on-hand
        self.BO = np.zeros(self.config.NoProducts)  # current backorder
        self.IT = np.zeros([self.config.NoComponents, max(self.config.L)])  # inventory in transit
        self.D = np.zeros(self.config.NoProducts)
        self.curTime = 0  # current time
        return self._get_observation()

    def seed(self, seed=None):
        pass

    def _apply_action(self, action):
        self.D = self._demand()
        # get actions
        action_O = action[:self.config.NoComponents]
        action_P = action[self.config.NoComponents:]
        # if self.curTime == 0:
        #     action_O = np.zeros(self.config.NoComponents)
        #     action_P = np.zeros(self.config.NoProducts)

        M = np.dot(action_P, self.config.usage_rate.T)
        M = np.array(M).flatten()

        arrival = self.IT[:, -1].flatten()
        self.IO = self.IO + arrival
        self.IO = self.IO - M
        self.BO = self.BO - action_P
        self.IT = np.roll(self.IT, 1)
        self.IT[:, [0]] = 0
        for i in range(self.config.NoComponents):
            self.IT[i, -self.config.L[i]] = action_O[i]
        self.BO = self.BO + self.D
        self.IO[self.IO < 0] = 0
        self.BO[self.BO < 0] = 0
        self.IT[self.IT < 0] = 0
        self.hitBO = (self.BO - self.config.BO_max).clip(0)
        self.hitIO = (self.IO - self.config.IO_max).clip(0)
        self.IO[self.IO > self.config.IO_max] = self.config.IO_max
        self.BO[self.BO > self.config.BO_max] = self.config.BO_max

    def _get_observation(self):
        IT_array = np.array([])
        for i in range(self.config.NoComponents):
            IT_array=np.append(IT_array,self.IT[i, -self.config.L[i]:])
        IT_array.flatten()
        state = np.concatenate((IT_array, self.IO, self.BO))
        state_norm = (state - self.config.observation_low) / (self.config.observation_high - self.config.observation_low)
        return state_norm

    def _demand(self):
        if self.config.demand_type == 0:
            demand = np.random.random(size=[self.config.NoProducts]) * self.config.demand_dist
        elif self.config.demand_type == 1:
            demand = np.array([np.random.poisson(self.config.demand_dist[i]) for i in range(self.config.NoProducts)])
        elif self.config.demand_type == 2:
            demand = np.array([gamma.rvs(self.config.demand_dist[0,i],self.config.demand_dist[1,i],self.config.demand_dist[2,i]) for i in range(self.config.NoProducts)])
        elif self.config.demand_type == 3:
            prop = 0.9
            demand = np.random.random(size=[self.config.NoProducts]) * self.config.demand_dist
            demand[0] = demand[0]
            tmp = random.random()
            if tmp<=prop:
                demand[1] = 40-demand[0]
        return demand

class ATOFixedLTCATrain(gym.Env):
    def __init__(self,conf):
        self.config = conf
        self.env_name = 'ATOCA-v0'
        self.state_dim = len(self.config.observation_low)  # feature number of state
        self.action_dim = len(self.config.action_low)  # feature number of action
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.max_step = 10000  # the max step of each episode
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = 1e15

        self.IO = np.zeros(self.config.NoComponents)  # current inventory on-hand
        self.BO = np.zeros(self.config.NoProducts)  # current backorder
        self.IT = np.zeros([self.config.NoComponents, max(self.config.L)])  # inventory in transit
        self.D = np.zeros(self.config.NoProducts)
        self.curTime = 0  # current time
        # 定义动作空间
        self.action_space = spaces.Box(
            high=self.config.action_high,
            low=self.config.action_low
        )

        # 定义状态空间
        self.observation_space = spaces.Box(
            high=self.config.observation_high,
            low=self.config.observation_low
        )

    def step(self, action):
        action = [(action[i] + 1) * self.config.action_high[i] / 2 for i in
                  range(self.config.NoProducts + self.config.NoComponents)]
        assert self.action_space.contains(action)
        self._apply_action(action)

        reward = -(sum(self.config.c_p[i] * (self.BO[i] - self.D[i]) for i in range(self.config.NoProducts))
                   + sum(self.config.c_h[i] * self.IO[i] for i in range(self.config.NoComponents))
                   + sum(self.config.c_h[i] * self.hitIO[i]/(1-0.95) for i in range(self.config.NoComponents))
                   + sum(self.config.c_p[i] * self.hitBO[i]/(1-0.95) for i in range(self.config.NoProducts)))

        self.curTime += 1
        if self.curTime == self.config.T:
            done = True
        else:
            done = False

        info = {}
        return self._get_observation(), float(reward), done, info

    def reset(self):
        self.IO = np.zeros(self.config.NoComponents)  # current inventory on-hand
        self.BO = np.zeros(self.config.NoProducts)  # current backorder
        self.IT = np.zeros([self.config.NoComponents, max(self.config.L)])  # inventory in transit
        self.D = np.zeros(self.config.NoProducts)
        self.curTime = 0  # current time
        return self._get_observation()

    def seed(self, seed=None):
        pass

    def _apply_action(self, action):
        self.D = self._demand()
        # get actions
        action_O = action[:self.config.NoComponents]
        action_P = action[self.config.NoComponents:]
        # if self.curTime == 0:
        #     action_O = np.zeros(self.config.NoComponents)
        #     action_P = np.zeros(self.config.NoProducts)

        M = np.dot(action_P, self.config.usage_rate.T)
        M = np.array(M).flatten()

        arrival = self.IT[:, -1].flatten()
        self.IO = self.IO + arrival
        self.IO = self.IO - M
        self.BO = self.BO - action_P
        self.IT = np.roll(self.IT, 1)
        self.IT[:, [0]] = 0
        for i in range(self.config.NoComponents):
            self.IT[i, -self.config.L[i]] = action_O[i]
        self.BO = self.BO + self.D
        self.IO[self.IO < 0] = 0
        self.BO[self.BO < 0] = 0
        self.IT[self.IT < 0] = 0
        self.hitBO = (self.BO - self.config.BO_max).clip(0)
        self.hitIO = (self.IO - self.config.IO_max).clip(0)
        self.IO[self.IO > self.config.IO_max] = self.config.IO_max
        self.BO[self.BO > self.config.BO_max] = self.config.BO_max

    def _apply_action_demand(self, action, demand):
        self.D = demand
        action_O = action[:self.config.NoComponents]
        action_P = action[self.config.NoComponents:]

        M = np.dot(action_P, self.config.usage_rate.T)
        M = np.array(M).flatten()

        arrival = self.IT[:, -1].flatten()
        self.IO = self.IO + arrival
        self.IO = self.IO - M
        self.BO = self.BO - action_P

        self.IT = np.roll(self.IT, 1, axis=1)
        self.IT[:, 0] = 0
        for i in range(self.config.NoComponents):
            self.IT[i, -self.config.L[i]] = action_O[i]

        self.BO = self.BO + self.D

        self.IO[self.IO < 0] = 0
        self.BO[self.BO < 0] = 0
        self.IT[self.IT < 0] = 0

        self.hitBO = (self.BO - self.config.BO_max).clip(0)
        self.hitIO = (self.IO - self.config.IO_max).clip(0)
        self.IO[self.IO > self.config.IO_max] = self.config.IO_max
        self.BO[self.BO > self.config.BO_max] = self.config.BO_max

    def _get_observation(self):
        IT_array = np.array([])
        for i in range(self.config.NoComponents):
            IT_array=np.append(IT_array,self.IT[i, -self.config.L[i]:])
        IT_array.flatten()
        state = np.concatenate((IT_array, self.IO, self.BO))
        state_norm = (state - self.config.observation_low) / (self.config.observation_high - self.config.observation_low)
        return state_norm

    def _demand(self):
        random_index = np.random.choice(self.config.demand_set.shape[0])
        demand = self.config.demand_set[random_index]
        return demand

    def step_with_demand(self, action, demand):
        return self._step_core(action, demand)

    def _step_core(self, action, demand):
        action = [(action[i] + 1) * self.config.action_high[i] / 2
                  for i in range(self.config.NoProducts + self.config.NoComponents)]
        assert self.action_space.contains(action)
        self._apply_action_demand(action, demand)

        reward = -(sum(self.config.c_p[i] * (self.BO[i] - self.D[i]) for i in range(self.config.NoProducts))
                   + sum(self.config.c_h[i] * self.IO[i] for i in range(self.config.NoComponents))
                   + sum(self.config.c_h[i] * self.hitIO[i] / (1 - 0.95) for i in range(self.config.NoComponents))
                   + sum(self.config.c_p[i] * self.hitBO[i] / (1 - 0.95) for i in range(self.config.NoProducts)))

        self.curTime += 1
        done = (self.curTime == self.config.T)
        return self._get_observation(), float(reward), done, {}
