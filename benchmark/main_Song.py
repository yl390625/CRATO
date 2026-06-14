import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from elegantrl.envs.ATO_env.ato_config import ATOTestConfig
from tqdm import tqdm
import math
from scipy.stats import norm, uniform
import gurobipy
from elegantrl.envs.ATO_env.ato_env import ATOFixedLTCATest
import argparse


def prp(conf, inventory_constr, backorder_constr):
    """
    PRP rule to determine the component allocation.
    """
    model = gurobipy.Model()
    model.setParam('OutputFlag', 0)
    B = model.addVars(range(conf.NoProducts), lb=0, vtype=gurobipy.GRB.CONTINUOUS)
    I = model.addVars(range(conf.NoComponents), lb=0, vtype=gurobipy.GRB.CONTINUOUS)
    model.update()

    # the optimization model
    model.ModelSense = gurobipy.GRB.MINIMIZE
    model.setObjective(sum(conf.c_p[j] * B[j] for j in range(conf.NoProducts)) + sum(
        conf.c_h[i] * I[i] for i in range(conf.NoComponents)))
    model.addConstrs(I[i] == inventory_constr[i]
                     - (np.array(conf.usage_rate[i]) * backorder_constr).sum()
                     + sum(conf.usage_rate[i, j] * B[j] for j in range(conf.NoProducts)) for i in
                     range(conf.NoComponents))
    model.addConstrs(inventory_constr[i]
                     - (np.array(conf.usage_rate[i]) * backorder_constr).sum()
                     + sum(conf.usage_rate[i, j] * B[j] for j in range(conf.NoProducts)) >= 0 for i in
                     range(conf.NoComponents))
    model.addConstrs(B[j] <= backorder_constr[j] for j in range(conf.NoProducts))
    model.optimize()
    B_hat = np.array([B[j].X for j in range(conf.NoProducts)])
    a = backorder_constr - B_hat
    a[a < 0] = 0
    return a

def nv_heuristic(conf, average, std):
    """
    Newsvendor heuristic from Song's article.
    """
    s = np.zeros(conf.NoComponents)
    for i in range(conf.NoComponents):
        avg_demand = (np.array(conf.usage_rate[i]) * np.array(average)).sum()
        std_demand = math.sqrt((np.array(conf.usage_rate[i]) * np.array(std) ** 2).sum())
        h = conf.c_h[i]
        b = (conf.c_p * np.array(conf.usage_rate[i]))
        b = b[b != 0].min()
        s[i] = avg_demand*conf.L[i]+norm.ppf(b/(b+h),0,std_demand)
    return s

def main():
    parser = argparse.ArgumentParser(description="NV-PRP heuristic.")
    parser.add_argument('--warmup', type=int, default=500, help='Warm-up period')
    parser.add_argument('--replication', type=int, default=100, help='Number of replications')
    parser.add_argument('--time_horizon', type=int, default=10000, help='Time horizon for each replication')
    parser.add_argument('--config_name', type=str, default='experimentN', help='Configuration filename')
    parser.add_argument('--sample_num', type=int, default=10, help='Number of demand samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    config_filename = args.config_name
    config_filepath = '../exp_data/' + config_filename + '.xlsx'
    num = args.sample_num
    warmup = args.warmup
    replication = args.replication
    time_horizon = args.time_horizon

    if config_filename == 'experimentN':
        demand = np.load(f'../data_generate/demand_sample/N_samples/num_samples_{num}.npy')
        conf = ATOTestConfig(config_filepath, order_max=800, product_max=800, BO_max=800, IO_max=800, demand_type=0)
    elif config_filename == 'experimentPC':
        demand = np.load(f'../data_generate/demand_sample/PC_samples/num_samples_{num}.npy')
        conf = ATOTestConfig(config_filepath, order_max=800, product_max=800, BO_max=800, IO_max=800, demand_type=1)
    elif config_filename == 'experimentLarge':
        demand = np.load(f'../data_generate/demand_sample/large_samples/num_samples.npy')
        conf = ATOTestConfig(config_filepath, order_max=800, product_max=800, BO_max=800, IO_max=800, demand_type=2)
    elif config_filename == 'experimentNCorr':
        demand = np.load(f'../data_generate/demand_sample/N_samples_corr/num_samples_{num}.npy')
        conf = ATOTestConfig(config_filepath, order_max=800, product_max=800, BO_max=800, IO_max=800, demand_type=3)

    average = demand.mean(0)
    std = demand.std(0, ddof=1)
    s = nv_heuristic(conf, average, std)

    env = ATOFixedLTCATest(conf)
    state = np.zeros(sum(conf.L) + conf.NoComponents + conf.NoProducts)
    action = np.zeros(conf.NoComponents + conf.NoProducts)
    rewards = np.zeros(replication)

    num_vars = conf.NoComponents + conf.NoProducts
    results = np.zeros((replication, time_horizon))
    a_max = []
    s_max = []
    for j in tqdm(range(replication)):
        reward = 0
        for i in range(time_horizon):
            index = [sum(conf.L[:p + 1]) - 1 for p in range(conf.NoComponents)]
            inventory_constr = state[sum(conf.L):-conf.NoProducts] + state[index]
            backorder_constr = state[-conf.NoProducts:]

            for k in range(conf.NoComponents):
                action[k] = max(s[k] - (inventory_constr[k] - np.multiply(conf.usage_rate[k], backorder_constr).sum()),
                                0)

            p_a = prp(conf, inventory_constr, backorder_constr)
            for k in range(conf.NoProducts):
                action[k + conf.NoComponents] = p_a[k]
            a_max.append(max(action))
            action = [2 * action[i] / conf.action_high[i] - 1 for i in range(num_vars)]
            s_max.append(np.max(state))
            state, r, _, _ = env.step(action=action)
            state = state.flatten() * (conf.observation_high - conf.observation_low) + conf.observation_low

            results[j, i] = r
            if i >= warmup:
                reward += r
        rewards[j] = reward / (time_horizon - warmup)

    print(rewards.mean(), rewards.std() / math.sqrt(replication))

if __name__ == '__main__':
    main()

