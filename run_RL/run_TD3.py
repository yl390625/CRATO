import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import gym
from elegantrl.train.run import *
from elegantrl.agents.ATO import AgentTD3_ATO
from elegantrl.train.config import Arguments
from elegantrl.envs.ATO_env.ato_env import ATOFixedLTCATrain
from elegantrl.envs.ATO_env.ato_config import ATOTrainConfig
import datetime
import argparse

gym.logger.set_level(40)  # Block warning

def main():
    parser = argparse.ArgumentParser(description="Train data-driven TD3 algorithm.")
    parser.add_argument('--exp_name', type=str, default='experimentN')
    parser.add_argument('--sample_num', type=str, default='10')
    parser.add_argument('--order_max', type=int, default=120)
    parser.add_argument('--product_max', type=int, default=150)
    parser.add_argument('--BO_max', type=int, default=500)
    parser.add_argument('--IO_max', type=int, default=500)
    parser.add_argument('--visible_gpu', type=int, default=0)
    parser.add_argument('--net_dim', type=int, default=2 ** 7)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    args = parser.parse_args()

    config_filename = args.exp_name
    config_filepath = '../exp_data/' + config_filename + '.xlsx'
    num = args.sample_num
    order_max = args.order_max
    product_max = args.product_max
    BO_max = args.BO_max
    IO_max = args.IO_max
    visible_gpu = args.visible_gpu
    net_dim = args.net_dim
    learning_rate = args.learning_rate


    if config_filename == 'experimentN':
        demand = np.load(f'../data_generate/demand_sample/N_samples/num_samples_{num}.npy')
    elif config_filename == 'experimentPC':
        demand = np.load(f'../data_generate/demand_sample/PC_samples/num_samples_{num}.npy')
    elif config_filename == 'experimentLarge':
        demand = np.load(f'../data_generate/demand_sample/large_samples/num_samples.npy')
    elif config_filename == 'experimentNCorr':
        demand = np.load(f'../data_generate/demand_sample/N_samples_corr/num_samples_{num}.npy')

    conf = ATOTrainConfig(config_filepath, order_max=order_max, product_max=product_max, BO_max=BO_max, IO_max=IO_max,
                          demand_type=0, demand_set=demand)
    env = ATOFixedLTCATrain(conf)
    agent = AgentTD3_ATO.AgentTD3_ATO
    rl_args = Arguments(agent, env, ATOConfig=conf)
    rl_args.visible_gpu = visible_gpu
    rl_args.net_dim = net_dim
    rl_args.learning_rate = learning_rate
    now = datetime.datetime.now()
    rl_args.cwd = f"models/TD3_{config_filename}_datasize{num}_date{now.strftime("%Y%m%d")}"

    train_and_evaluate(rl_args)



if __name__ == '__main__':
    main()
