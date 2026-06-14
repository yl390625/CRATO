from train_dp import ValueIterationDPTest, test_dp_policy
from elegantrl.envs.ATO_env.ato_config import ATOTestConfig
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate DP algorithm.")
    parser.add_argument('--exp_name', type=str, default='experimentN')
    parser.add_argument('--policy_name', type=str, default='dp_policy_20260115_220742')
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--num_episodes', type=int, default=100)

    args = parser.parse_args()
    policy_path = f'models/{args.policy_name}.pkl'

    config_filename = args.exp_name
    config_filepath = '../exp_data/' + config_filename + '.xlsx'
    episode_length = args.episode_length

    conf = ATOTestConfig(config_filepath, order_max=600, product_max=600,
                     BO_max=800, IO_max=800, demand_type=0)

    dp_agent = ValueIterationDPTest(conf, gamma=0.95, theta=1e-4, max_iter=1000)

    dp_agent.load_policy(policy_path)
    dp_agent.config = conf

    test_dp_policy(dp_agent, num_episodes=args.num_episodes, episode_length=episode_length)

if __name__ == '__main__':
    main()