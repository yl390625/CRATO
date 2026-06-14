import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from elegantrl.train_ICNN.evaluator import *
from elegantrl.agents.ATO_ICNN import AgentTD3_ATO_ICNN
from elegantrl.train_ICNN.config import Arguments
from elegantrl.envs.ATO_env.ato_env_ICNN import ATOFixedLTCATest
from tqdm import tqdm
import pandas as pd
from elegantrl.envs.ATO_env.ato_config import ATOTestConfig
import math
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate CTD3 agents. Order/product/BO/IO_max need to be aligned with training settings.")
    parser.add_argument('--exp_name', type=str, default='experimentN')
    parser.add_argument('--model_names', nargs='+', default=['num10'], help='List of model names to evaluate')
    parser.add_argument('--warmup', type=int, default=500, help='Warm-up period')
    parser.add_argument('--replication', type=int, default=100, help='Number of replications')
    parser.add_argument('--time_horizon', type=int, default=10000, help='Time horizon for each replication')
    parser.add_argument('--visible_gpu', type=int, default=0)
    parser.add_argument('--net_dim', type=int, default=2 ** 7)
    parser.add_argument('--order_max', type=int, default=120)
    parser.add_argument('--product_max', type=int, default=150)
    parser.add_argument('--BO_max', type=int, default=500)
    parser.add_argument('--IO_max', type=int, default=500)

    args = parser.parse_args()

    config_filename = args.exp_name
    T = args.time_horizon
    warmup = args.warmup
    replication = args.replication
    visible_gpu = args.visible_gpu
    net_dim = args.net_dim
    order_max = args.order_max
    product_max = args.product_max
    BO_max = args.BO_max
    IO_max = args.IO_max

    config_filepath = '../exp_data/' + config_filename + '.xlsx'

    if config_filename == 'experimentN':
        conf = ATOTestConfig(config_filepath, order_max=order_max, product_max=product_max, BO_max=BO_max, IO_max=IO_max, demand_type=0, T=T)
    elif config_filename == 'experimentPC':
        conf = ATOTestConfig(config_filepath, order_max=order_max, product_max=product_max, BO_max=BO_max, IO_max=IO_max, demand_type=1, T=T)
    elif config_filename == 'experimentLarge':
        conf = ATOTestConfig(config_filepath, order_max=order_max, product_max=product_max, BO_max=BO_max, IO_max=IO_max, demand_type=2, T=T)
    elif config_filename == 'experimentNCorr':
        conf = ATOTestConfig(config_filepath, order_max=order_max, product_max=product_max, BO_max=BO_max, IO_max=IO_max, demand_type=3, T=T)

    r_avg_arr = []
    r_std_arr = []
    for name in args.model_names:
        a, s = evaluator_actor_pth(name, conf, T, warmup, replication, visible_gpu, net_dim)
        r_avg_arr.append(a)
        r_std_arr.append(s)
    r_avg_arr = np.array(r_avg_arr)
    r_std_arr = np.array(r_std_arr)

    # Create a DataFrame object to store the results
    df = pd.DataFrame({
        'Model Name': args.model_names,
        'Average Returns': r_avg_arr,
        'Standard Deviations': r_std_arr
    })

    # Save the DataFrame to a CSV file
    df.to_csv(f'results/{config_filename}_CTD3_results.csv', index=False)

def evaluator_actor_pth(name, conf, T, warmup, replication, visible_gpu, net_dim):
    actor_path = '../run_RL/models/'+name
    T = T
    warmup = warmup
    replication = replication
    env = ATOFixedLTCATest(conf)

    gpu_id = int(visible_gpu)

    agt = AgentTD3_ATO_ICNN

    args = Arguments(agt, env, ATOConfig=conf)
    args.cwd = 'TD3'
    args.visible_gpu = visible_gpu

    """init"""
    agent = AgentTD3_ATO_ICNN.AgentTD3_ATO_ICNN(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id,args=args)
    cwd = actor_path
    agent.save_or_load_agent(cwd, if_save=False)

    """evaluate"""
    r_s_ary = [get_episode_return_and_step(env, agent.act,conf,warmup) for _ in tqdm(range(replication))]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)[:,0]/(T-warmup)
    r_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    r_std = r_s_ary.std(axis=0)

    print("name:", name)
    print(f"r_avg: {r_avg},  r_std: {r_std / math.sqrt(replication)}")
    return r_avg, r_std

if __name__ == "__main__":
    main()
