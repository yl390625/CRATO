import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from tqdm import tqdm
from elegantrl.envs.ATO_env.ato_config import ATOTestConfig
from elegantrl.envs.ATO_env.ato_env import ATOFixedLTCATest
from scipy.optimize import fsolve
import scipy.integrate as integrate


# Experiment Configurations
config_filename = 'experimentN'
config_filepath = '../exp_data/' + config_filename + '.xlsx'
conf = ATOTestConfig(config_filepath, order_max=200, product_max=200, BO_max=500, IO_max=500, demand_type=0)

# N-system parameters
a = conf.demand_dist[0]  # Upper bound for D^1
b = conf.demand_dist[1]  # Upper bound for D^12
c = conf.c_p[0]  # Given parameter
h1 = conf.c_h[0]  # Given parameter
h2 = conf.c_h[1]  # Given parameter


def equations(vars):
    s1, s2 = vars

    def P_30_integrand1(x):
        return b - (s1 - x)

    def P_30_integrand2(x):
        return b

    x1_end = min(a, s1)
    P_30_part1, _ = integrate.quad(P_30_integrand1, s1 - s2, x1_end)
    P_30_part2, _ = integrate.quad(P_30_integrand2, x1_end, a)
    P_30 = (P_30_part1 + P_30_part2) / (a * b)

    P_31 = ((b - s2) * (s1 - s2)) / (a * b)

    eq1 = P_30 - (h1 / (c + h1))
    eq2 = P_31 - (h2 / (c + h1))

    return [eq1, eq2]


if __name__ == '__main__':
    warmup = 500
    replication = 100
    time_horizon = 10000

    initial_guess = [5, 5]

    solution = fsolve(equations, initial_guess)

    s = solution
    state = np.zeros(6)
    action = np.zeros(4)

    env = ATOFixedLTCATest(conf)
    num_vars = conf.NoComponents + conf.NoProducts
    rewards = np.zeros(replication)
    for j in tqdm(range(replication)):
        reward = 0
        for i in range(time_horizon):
            if state[0]+state[2]-state[4]-state[5] < s[0]:
                action[0] = s[0] - state[0]-state[2]+state[4]+state[5]
            else:
                action[0] = 0
            if state[1]+state[3]-state[5] < s[1]:
                action[1] = s[1] - state[1]-state[3]+state[5]
            else:
                action[1] = 0
            action[2] = min(state[0]+state[2],state[4])
            action[3] = min(min(state[0]+state[2]-action[2],state[1]+state[3]),state[5])
            action=[2 * action[k] / conf.action_high[k] - 1 for k in range(num_vars)]
            state,r,_,_ = env.step(action = action)
            state = state*(conf.observation_high-conf.observation_low)+conf.observation_low
            if i >= warmup:
                reward += r
        rewards[j] = reward / (time_horizon - warmup)
    print(f"Optimal simulated average cost: {rewards.mean()}")
