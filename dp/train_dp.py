import math

import numpy as np
import gym, logging
from elegantrl.envs.ATO_env.ato_config import ATOTrainConfig
from elegantrl.envs.ATO_env.ato_env import ATOFixedLTCATrain, ATOFixedLTCATest
import gurobipy
import pickle
import datetime
import os
from tqdm import tqdm
from itertools import product
gym.logger.set_level(logging.ERROR)
import argparse


ACTION_LOW = 0
ACTION_HIGH = 80
STATE_LOW = 0
STATE_HIGH = 80
ACTION_NUM = 4
STATE_NUM = 4

class EmpiricalDemandModel:
    def __init__(self, demand_set: np.ndarray):
        uniq, counts = np.unique(demand_set, axis=0, return_counts=True)
        self.scenarios = uniq
        self.probs = counts / counts.sum()

    def iter(self):
        for d, p in zip(self.scenarios, self.probs):
            yield d, p

class ValueIterationDP:
    def __init__(self, config, gamma=0.95, theta=1e-6, max_iter=1000):
        self.config = config
        self.gamma = gamma
        self.theta = theta
        self.max_iter = max_iter
        self.pipeline_len = int(sum(self.config.L))
        self.prp_cache = {}

        self.setup_discrete_spaces()

        self.V = np.zeros(self.state_space_size)
        self.policy = np.zeros((self.state_space_size, self.action_space_size))

        self.env = ATOFixedLTCATrain(config)

        self.state_index_map = {comb: i for i, comb in enumerate(self.state_combinations)}
        self.demand_model = EmpiricalDemandModel(self.config.demand_set)

    def setup_discrete_spaces(self):
        self.order_actions = np.linspace(ACTION_LOW, ACTION_HIGH, ACTION_NUM)
        self.state_levels = np.linspace(STATE_LOW, STATE_HIGH, STATE_NUM)

        state_dims = self.pipeline_len + self.config.NoComponents + self.config.NoProducts

        self.action_space_size = len(self.order_actions) ** self.config.NoComponents
        self.state_space_size = len(self.state_levels) ** state_dims

        self.action_combinations = list(product(range(len(self.order_actions)), repeat=self.config.NoComponents))
        self.state_combinations = list(product(range(len(self.state_levels)), repeat=state_dims))

    def continuous_to_discrete_state(self, continuous_state):
        indices = []

        def val_to_idx(v):
            v_clamped = min(max(v, STATE_LOW), STATE_HIGH)
            ratio = (v_clamped - STATE_LOW) / (STATE_HIGH - STATE_LOW)
            return int(round(ratio * (STATE_NUM - 1)))

        # IT
        for comp in range(self.config.NoComponents):
            Li = self.config.L[comp]
            base = sum(self.config.L[:comp])
            for k in range(Li):
                indices.append(val_to_idx(continuous_state[base + k]))
        # IO
        io_start = sum(self.config.L)
        for i in range(self.config.NoComponents):
            indices.append(val_to_idx(continuous_state[io_start + i]))
        # BO
        bo_start = io_start + self.config.NoComponents
        for j in range(self.config.NoProducts):
            indices.append(val_to_idx(continuous_state[bo_start + j]))
        return tuple(indices)

    def discrete_to_continuous_state(self, discrete_state_idx):
        comb = self.state_combinations[discrete_state_idx]
        cont = np.zeros(len(self.config.observation_low))
        # IT segment
        cursor = 0
        for comp in range(self.config.NoComponents):
            Li = self.config.L[comp]
            for k in range(Li):
                level_idx = comb[cursor]
                cont_idx = sum(self.config.L[:comp]) + k
                cont[cont_idx] = self.state_levels[level_idx]
                cursor += 1
        # IO
        io_start = sum(self.config.L)
        for i in range(self.config.NoComponents):
            level_idx = comb[cursor]
            cont[io_start + i] = self.state_levels[level_idx]
            cursor += 1
        # BO
        bo_start = io_start + self.config.NoComponents
        for j in range(self.config.NoProducts):
            level_idx = comb[cursor]
            cont[bo_start + j] = self.state_levels[level_idx]
            cursor += 1
        return cont

    def discrete_to_continuous_action(self, discrete_action_idx):
        action_combination = self.action_combinations[discrete_action_idx]
        continuous_action = np.zeros(self.config.NoComponents + self.config.NoProducts)

        for i in range(self.config.NoComponents):
            level_idx = action_combination[i]
            continuous_action[i] = self.order_actions[level_idx]

        return continuous_action

    def get_cached_prp_allocation(self, state_idx, continuous_state=None):
        if state_idx in self.prp_cache:
            return self.prp_cache[state_idx]

        if continuous_state is None:
            continuous_state = self.discrete_to_continuous_state(state_idx)

        allocation = self.nhb_allocation(continuous_state)  # 或 prp_allocation(...)
        self.prp_cache[state_idx] = allocation
        return allocation

    def nhb_allocation(self, state):
        allocation = np.zeros(self.config.NoProducts)
        allocation[0] = min(state[0] + state[2], state[4])
        allocation[1] = min(min(state[0] + state[2] - allocation[0], state[1] + state[3]), state[5])
        return allocation

    def prp_allocation(self, inventory_constr, backorder_constr):
        model = gurobipy.Model()
        model.setParam('OutputFlag', 0)
        B = model.addVars(range(self.config.NoProducts), lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        I = model.addVars(range(self.config.NoComponents), lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        model.update()

        model.ModelSense = gurobipy.GRB.MINIMIZE
        model.setObjective(sum(self.config.c_p[j] * B[j] for j in range(self.config.NoProducts)) + sum(
            self.config.c_h[i] * I[i] for i in range(self.config.NoComponents)))

        model.addConstrs(I[i] == inventory_constr[i]
                         - (np.array(self.config.usage_rate[i]) * backorder_constr).sum()
                         + sum(self.config.usage_rate[i, j] * B[j] for j in range(self.config.NoProducts)) for i in
                         range(self.config.NoComponents))

        model.addConstrs(inventory_constr[i]
                         - (np.array(self.config.usage_rate[i]) * backorder_constr).sum()
                         + sum(self.config.usage_rate[i, j] * B[j] for j in range(self.config.NoProducts)) >= 0 for i in
                         range(self.config.NoComponents))

        model.addConstrs(B[j] <= backorder_constr[j] for j in range(self.config.NoProducts))

        try:
            model.optimize()
            if model.status == gurobipy.GRB.OPTIMAL:
                B_hat = np.array([B[j].X for j in range(self.config.NoProducts)])
                allocation = backorder_constr - B_hat
                allocation[allocation < 0] = 0
                return allocation
            else:
                return np.zeros(self.config.NoProducts)
        except:
            return np.zeros(self.config.NoProducts)

    def _precompute_all(self):
        S = self.state_space_size
        A = self.action_space_size
        demand_scenarios = np.array([d for d, _ in self.demand_model.iter()])
        demand_probs = np.array([p for _, p in self.demand_model.iter()])
        R = np.zeros((S, A), dtype=np.float32)
        P = [[dict() for _ in range(A)] for _ in range(S)]
        cont_states = [self.discrete_to_continuous_state(s) for s in range(S)]

        def rebuild_IT_matrix(pipeline_vector):
            IT = np.zeros((self.config.NoComponents, max(self.config.L)))
            cursor = 0
            for comp in range(self.config.NoComponents):
                Li = self.config.L[comp]
                segment = pipeline_vector[cursor:cursor + Li]
                IT[comp, -Li:] = segment
                cursor += Li
            return IT

        def extract_pipeline(IT_mat):
            pieces = []
            for comp in range(self.config.NoComponents):
                Li = self.config.L[comp]
                pieces.append(IT_mat[comp, -Li:])
            return np.concatenate(pieces, axis=0)

        def process_state(s):
            state_res_R = np.zeros(A, dtype=np.float32)
            state_res_P = [dict() for _ in range(A)]
            base_cont = cont_states[s]

            io_start = sum(self.config.L)
            io_end = io_start + self.config.NoComponents
            bo_start = io_end

            pipeline_vec = base_cont[:self.pipeline_len]

            for a in range(A):
                ca = self.discrete_to_continuous_action(a)
                prp = self.get_cached_prp_allocation(s, base_cont)
                full_action = np.concatenate([ca[:self.config.NoComponents], prp])
                norm_action = 2 * full_action / self.config.action_high[:len(full_action)] - 1
                exp_r = 0.0
                trans_acc = {}

                for d, p in zip(demand_scenarios, demand_probs):
                    self.env.reset()
                    self.env.IT = rebuild_IT_matrix(pipeline_vec).copy()
                    self.env.IO = base_cont[io_start:io_end].copy()
                    self.env.BO = base_cont[bo_start:].copy()
                    self.env.D = d.copy()

                    _, r, _, _ = self.env.step_with_demand(norm_action, d)

                    raw_state = np.zeros(len(self.config.observation_low))
                    # IT
                    raw_pipeline = extract_pipeline(self.env.IT)
                    raw_state[:self.pipeline_len] = raw_pipeline
                    # IO
                    raw_state[io_start:io_end] = self.env.IO
                    # BO
                    raw_state[bo_start:] = self.env.BO

                    disc_next = self.continuous_to_discrete_state(raw_state)
                    next_idx = self.state_index_map.get(disc_next, s)
                    exp_r += p * r
                    trans_acc[next_idx] = trans_acc.get(next_idx, 0.0) + p

                state_res_R[a] = exp_r
                state_res_P[a] = trans_acc
            return s, state_res_R, state_res_P

        for s in tqdm(range(S)):
            s, r_row, p_row = process_state(s)
            R[s] = r_row
            P[s] = p_row


        self.precomputed_R = R
        self.precomputed_P = P

    def value_iteration(self):
        if not hasattr(self, 'precomputed_R'):
            self._precompute_all()
        R = self.precomputed_R
        P = self.precomputed_P
        for it in tqdm(range(self.max_iter)):
            delta = 0.0
            newV = np.zeros_like(self.V)
            for s in range(self.state_space_size):
                q_vals = []
                for a in range(self.action_space_size):
                    future = 0.0
                    for sn, prob in P[s][a].items():
                        future += prob * self.V[sn]
                    q_vals.append(R[s, a] + self.gamma * future)
                q_vals = np.array(q_vals)
                best = q_vals.max()
                newV[s] = best
                self.policy[s] = 0
                self.policy[s, q_vals.argmax()] = 1
                delta = max(delta, abs(self.V[s] - best))
            self.V = newV
            if delta < self.theta:
                break
        return self.V, self.policy


    def get_action(self, state):
        disc = self.continuous_to_discrete_state(state)
        state_idx = self.state_index_map.get(disc, 0)
        best_a = np.argmax(self.policy[state_idx])
        return self.discrete_to_continuous_action(best_a)

    def save_policy(self, filename):
        policy_data = {
            'V': self.V,
            'policy': self.policy,
            'action_combinations': self.action_combinations,
            'state_combinations': self.state_combinations,
            'order_actions': self.order_actions,
            'state_levels': self.state_levels
        }

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(policy_data, f)
        print(f"Policy saved to: {filename}")

    def load_policy(self, filename):
        with open(filename, 'rb') as f:
            policy_data = pickle.load(f)

        self.V = policy_data['V']
        self.policy = policy_data['policy']
        self.action_combinations = policy_data['action_combinations']
        self.state_combinations = policy_data['state_combinations']
        self.order_actions = policy_data['order_actions']
        self.state_levels = policy_data['state_levels']
        print(f"Policy loaded from: {filename}")

class ValueIterationDPTest:
    def __init__(self, config, gamma=0.95, theta=1e-6, max_iter=1000):
        self.config = config
        self.gamma = gamma
        self.theta = theta
        self.max_iter = max_iter
        self.pipeline_len = int(sum(self.config.L))
        self.prp_cache = {}

        self.setup_discrete_spaces()

        self.V = np.zeros(self.state_space_size)
        self.policy = np.zeros((self.state_space_size, self.action_space_size))

        self.env = ATOFixedLTCATest(config)

        self.state_index_map = {comb: i for i, comb in enumerate(self.state_combinations)}

    def setup_discrete_spaces(self):
        self.order_actions = np.linspace(ACTION_LOW, ACTION_HIGH, ACTION_NUM)
        self.state_levels = np.linspace(STATE_LOW, STATE_HIGH, STATE_NUM)

        state_dims = self.pipeline_len + self.config.NoComponents + self.config.NoProducts

        self.action_space_size = len(self.order_actions) ** self.config.NoComponents
        self.state_space_size = len(self.state_levels) ** state_dims


        self.action_combinations = list(product(range(len(self.order_actions)), repeat=self.config.NoComponents))
        self.state_combinations = list(product(range(len(self.state_levels)), repeat=state_dims))

    def continuous_to_discrete_state(self, continuous_state):
        indices = []

        def val_to_idx(v):
            v_clamped = min(max(v, STATE_LOW), STATE_HIGH)
            ratio = (v_clamped - STATE_LOW) / (STATE_HIGH - STATE_LOW)
            return int(round(ratio * (STATE_NUM - 1)))

        # IT
        for comp in range(self.config.NoComponents):
            Li = self.config.L[comp]
            base = sum(self.config.L[:comp])
            for k in range(Li):
                indices.append(val_to_idx(continuous_state[base + k]))
        # IO
        io_start = sum(self.config.L)
        for i in range(self.config.NoComponents):
            indices.append(val_to_idx(continuous_state[io_start + i]))
        # BO
        bo_start = io_start + self.config.NoComponents
        for j in range(self.config.NoProducts):
            indices.append(val_to_idx(continuous_state[bo_start + j]))
        return tuple(indices)

    def discrete_to_continuous_state(self, discrete_state_idx):
        comb = self.state_combinations[discrete_state_idx]
        cont = np.zeros(len(self.config.observation_low))
        # IT segment
        cursor = 0
        for comp in range(self.config.NoComponents):
            Li = self.config.L[comp]
            for k in range(Li):
                level_idx = comb[cursor]
                cont_idx = sum(self.config.L[:comp]) + k
                cont[cont_idx] = self.state_levels[level_idx]
                cursor += 1
        # IO
        io_start = sum(self.config.L)
        for i in range(self.config.NoComponents):
            level_idx = comb[cursor]
            cont[io_start + i] = self.state_levels[level_idx]
            cursor += 1
        # BO
        bo_start = io_start + self.config.NoComponents
        for j in range(self.config.NoProducts):
            level_idx = comb[cursor]
            cont[bo_start + j] = self.state_levels[level_idx]
            cursor += 1
        return cont

    def discrete_to_continuous_action(self, discrete_action_idx):
        action_combination = self.action_combinations[discrete_action_idx]
        continuous_action = np.zeros(self.config.NoComponents + self.config.NoProducts)

        for i in range(self.config.NoComponents):
            level_idx = action_combination[i]
            continuous_action[i] = self.order_actions[level_idx]

        return continuous_action

    def get_cached_prp_allocation(self, state_idx, continuous_state=None):
        # 需要构造代表性连续状态
        if continuous_state is None:
            continuous_state = self.discrete_to_continuous_state(state_idx)

        io_start = sum(self.config.L)
        io_end = io_start + self.config.NoComponents
        bo_start = io_end

        PI = continuous_state[:io_start].copy()
        IO = continuous_state[io_start:io_end].copy()
        BO = continuous_state[bo_start:].copy()

        index = [sum(self.config.L[:p + 1]) - 1 for p in range(self.config.NoComponents)]
        pipeline_est = PI

        inventory_constr = IO + pipeline_est
        backorder_constr = BO

        allocation = self.prp_allocation(inventory_constr, backorder_constr)
        return allocation

    def nhb_allocation(self, state):
        allocation = np.zeros(self.config.NoProducts)
        allocation[0] = min(state[0] + state[2], state[4])
        allocation[1] = min(min(state[0] + state[2] - allocation[0], state[1] + state[3]), state[5])
        return allocation

    def prp_allocation(self, inventory_constr, backorder_constr):
        model = gurobipy.Model()
        model.setParam('OutputFlag', 0)
        B = model.addVars(range(self.config.NoProducts), lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        I = model.addVars(range(self.config.NoComponents), lb=0, vtype=gurobipy.GRB.CONTINUOUS)
        model.update()

        model.ModelSense = gurobipy.GRB.MINIMIZE
        model.setObjective(sum(self.config.c_p[j] * B[j] for j in range(self.config.NoProducts)) + sum(
            self.config.c_h[i] * I[i] for i in range(self.config.NoComponents)))

        model.addConstrs(I[i] == inventory_constr[i]
                         - (np.array(self.config.usage_rate[i]) * backorder_constr).sum()
                         + sum(self.config.usage_rate[i, j] * B[j] for j in range(self.config.NoProducts)) for i in
                         range(self.config.NoComponents))

        model.addConstrs(inventory_constr[i]
                         - (np.array(self.config.usage_rate[i]) * backorder_constr).sum()
                         + sum(self.config.usage_rate[i, j] * B[j] for j in range(self.config.NoProducts)) >= 0 for i in
                         range(self.config.NoComponents))

        model.addConstrs(B[j] <= backorder_constr[j] for j in range(self.config.NoProducts))

        model.optimize()
        if model.status == gurobipy.GRB.OPTIMAL:
            B_hat = np.array([B[j].X for j in range(self.config.NoProducts)])
            allocation = backorder_constr - B_hat
            allocation[allocation < 0] = 0
            return allocation


    def _precompute_all(self):
        S = self.state_space_size
        A = self.action_space_size
        demand_scenarios = np.array([d for d, _ in self.demand_model.iter()])
        demand_probs = np.array([p for _, p in self.demand_model.iter()])
        R = np.zeros((S, A), dtype=np.float32)
        P = [[dict() for _ in range(A)] for _ in range(S)]
        cont_states = [self.discrete_to_continuous_state(s) for s in range(S)]

        def rebuild_IT_matrix(pipeline_vector):
            IT = np.zeros((self.config.NoComponents, max(self.config.L)))
            cursor = 0
            for comp in range(self.config.NoComponents):
                Li = self.config.L[comp]
                segment = pipeline_vector[cursor:cursor + Li]
                IT[comp, -Li:] = segment
                cursor += Li
            return IT

        def extract_pipeline(IT_mat):
            pieces = []
            for comp in range(self.config.NoComponents):
                Li = self.config.L[comp]
                pieces.append(IT_mat[comp, -Li:])
            return np.concatenate(pieces, axis=0)

        def process_state(s):
            state_res_R = np.zeros(A, dtype=np.float32)
            state_res_P = [dict() for _ in range(A)]
            base_cont = cont_states[s]

            # 预取索引
            io_start = sum(self.config.L)
            io_end = io_start + self.config.NoComponents
            bo_start = io_end

            # 拆分 pipeline 段
            pipeline_vec = base_cont[:self.pipeline_len]

            for a in range(A):
                ca = self.discrete_to_continuous_action(a)
                prp = self.get_cached_prp_allocation(s, base_cont)
                full_action = np.concatenate([ca[:self.config.NoComponents], prp])
                norm_action = 2 * full_action / self.config.action_high[:len(full_action)] - 1
                exp_r = 0.0
                trans_acc = {}

                for d, p in zip(demand_scenarios, demand_probs):
                    self.env.reset()
                    # 重建 IT / IO / BO
                    self.env.IT = rebuild_IT_matrix(pipeline_vec).copy()
                    self.env.IO = base_cont[io_start:io_end].copy()
                    self.env.BO = base_cont[bo_start:].copy()
                    self.env.D = d.copy()

                    _, r, _, _ = self.env.step_with_demand(norm_action, d)

                    # 组装下一原始状态
                    raw_state = np.zeros(len(self.config.observation_low))
                    # IT
                    raw_pipeline = extract_pipeline(self.env.IT)
                    raw_state[:self.pipeline_len] = raw_pipeline
                    # IO
                    raw_state[io_start:io_end] = self.env.IO
                    # BO
                    raw_state[bo_start:] = self.env.BO

                    disc_next = self.continuous_to_discrete_state(raw_state)
                    next_idx = self.state_index_map.get(disc_next, s)
                    exp_r += p * r
                    trans_acc[next_idx] = trans_acc.get(next_idx, 0.0) + p

                state_res_R[a] = exp_r
                state_res_P[a] = trans_acc
            return s, state_res_R, state_res_P

        print("预计算转移与回报(含 IT)...")

        for s in tqdm(range(S)):
            s, r_row, p_row = process_state(s)
            R[s] = r_row
            P[s] = p_row

        self.precomputed_R = R
        self.precomputed_P = P

    def value_iteration(self):
        if not hasattr(self, 'precomputed_R'):
            self._precompute_all()
        R = self.precomputed_R
        P = self.precomputed_P
        for it in tqdm(range(self.max_iter)):
            delta = 0.0
            newV = np.zeros_like(self.V)
            for s in range(self.state_space_size):
                q_vals = []
                for a in range(self.action_space_size):
                    future = 0.0
                    for sn, prob in P[s][a].items():
                        future += prob * self.V[sn]
                    q_vals.append(R[s, a] + self.gamma * future)
                q_vals = np.array(q_vals)
                best = q_vals.max()
                newV[s] = best
                self.policy[s] = 0
                self.policy[s, q_vals.argmax()] = 1
                delta = max(delta, abs(self.V[s] - best))
            self.V = newV
            if delta < self.theta:
                break
        return self.V, self.policy

    def get_action(self, state):
        # 若传入为环境归一化观测, 需先反归一化(此处假设外部已提供未归一化, 保持原逻辑)
        disc = self.continuous_to_discrete_state(state)
        state_idx = self.state_index_map.get(disc, 0)
        best_a = np.argmax(self.policy[state_idx])
        return self.discrete_to_continuous_action(best_a)

    def save_policy(self, filename):
        policy_data = {
            'V': self.V,
            'policy': self.policy,
            'action_combinations': self.action_combinations,
            'state_combinations': self.state_combinations,
            'order_actions': self.order_actions,
            'state_levels': self.state_levels
        }

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(policy_data, f)

    def load_policy(self, filename):
        with open(filename, 'rb') as f:
            policy_data = pickle.load(f)

        self.V = policy_data['V']
        self.policy = policy_data['policy']
        self.action_combinations = policy_data['action_combinations']
        self.state_combinations = policy_data['state_combinations']
        self.order_actions = policy_data['order_actions']
        self.state_levels = policy_data['state_levels']
        print(f"Policy loaded from {filename}")


def run_dp_experiment():
    parser = argparse.ArgumentParser(description="Train DP benchmark.")
    parser.add_argument('--sample_num', type=int, default=10, help='Number of demand samples')
    parser.add_argument('--exp_name', type=str, default='experimentN')
    args = parser.parse_args()

    config_filename = args.exp_name
    config_filepath = '../exp_data/' + config_filename + '.xlsx'
    demand_sample_name = f'num_samples_{args.sample_num}'
    if config_filename == 'experimentN':
        demand_set = np.load('../data_generate/demand_sample/N_samples/' + demand_sample_name + '.npy')
    else:
        demand_set = np.load('../data_generate/demand_sample/N_samples_corr/' + demand_sample_name + '.npy')

    conf = ATOTrainConfig(config_filepath, order_max=600, product_max=600,
                     BO_max=800, IO_max=800, demand_type=2, demand_set=demand_set)

    dp_agent = ValueIterationDP(conf, gamma=0.95, theta=1e-4, max_iter=1000)

    V, policy = dp_agent.value_iteration()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_filename = f'models/dp_policy_{timestamp}.pkl'
    dp_agent.save_policy(policy_filename)

    return dp_agent

def denorm_state(state_norm, conf):
    return state_norm * (conf.observation_high - conf.observation_low) + conf.observation_low

def test_dp_policy(dp_agent, num_episodes=10, episode_length=100):
    conf = dp_agent.config
    env = ATOFixedLTCATest(conf)
    total_rewards = []
    for episode in range(num_episodes):
        state_norm = env.reset()
        state_raw = denorm_state(state_norm, conf)
        episode_reward = 0.0

        for step in range(episode_length):
            continuous_action_raw = dp_agent.get_action(state_raw)

            io_start = sum(conf.L)
            io_end = io_start + conf.NoComponents
            bo_start = io_end

            disc_state = dp_agent.continuous_to_discrete_state(state_raw)
            state_idx = dp_agent.state_index_map.get(disc_state, 0)
            product_allocation = dp_agent.get_cached_prp_allocation(state_idx, state_raw)

            full_action_raw = np.concatenate([
                continuous_action_raw[:conf.NoComponents],
                product_allocation
            ])

            norm_action = 2 * full_action_raw / conf.action_high[:len(full_action_raw)] - 1

            next_state_norm, reward, done, _ = env.step(norm_action)
            episode_reward += reward

            state_raw = denorm_state(next_state_norm, conf)


        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward={episode_reward/episode_length:.2f}")

    print(f"Average cost: {np.mean(total_rewards)/episode_length:.2f}")
    return total_rewards


if __name__ == '__main__':
    run_dp_experiment()

