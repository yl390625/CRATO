from elegantrl.agents.ATO.AgentTD3_ATO import AgentTD3_ATO
import gym
from elegantrl.train.run import *
from elegantrl.train.evaluator import *
from elegantrl.agents.ATO import AgentTD3_ATO
from elegantrl.train.config import Arguments
from elegantrl.envs.ATO_env.ato_env import ATOFixedLTCA


def evaluator_actor_pth():
    gpu_id = 0  # >=0 means GPU ID, -1 means CPU

    agent = AgentTD3_ATO

    env = ATOFixedLTCA()
    agent = AgentTD3_ATO.AgentTD3_ATO
    args = Arguments(agent, env)
    args.cwd = 'TD3'
    args.visible_gpu = '0'

    actor_path = "./LunarLanderContinuous-v2_PPO_1/actor.pth"
    eval_times = 1000
    net_dim = 2**7

    """init"""
    act = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act
    act.load_state_dict(
        torch.load(actor_path, map_location=lambda storage, loc: storage)
    )

    """evaluate"""
    r_s_ary = [get_episode_return_and_step(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step

    print("r_avg, s_avg", r_avg, s_avg)
    return r_avg, s_avg

if __name__ == "__main__":
    # demo_evaluate_actors()
    evaluate_actors()
