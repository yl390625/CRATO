import torch
from elegantrl.agents.ATO_ICNN.net_ATO_ICNN import ActorICNN, CriticTwinICNN
from elegantrl.agents.ATO_ICNN.AgentDDPG_ATO_ICNN import AgentDDPG_ATO_ICNN
import numpy as np
from elegantrl.envs.ATO_env.gym_utils import projection_function, penalty_func

class weightConstraint():
    '''
        对一个特定的层的weight进行参数限制
    '''

    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0,1e15)
            module.weight.data = w

class AgentTD3_ATO_ICNN(AgentDDPG_ATO_ICNN):
    """
    Bases: ``AgentBase``

    Twin Delayed DDPG algorithm. “Addressing Function Approximation Error in Actor-Critic Methods”. Scott Fujimoto. et al.. 2015.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, "act_class", ActorICNN)
        self.cri_class = getattr(self, "cri_class", CriticTwinICNN)
        self.config = args.ATOConfig
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.policy_noise = getattr(
            args, "policy_noise", 0
        )  # standard deviation of policy noise
        self.update_freq = getattr(args, "update_freq", 2)  # delay update frequency
        self.beta_1 = 1e-5
        self.beta_2 = 1000

    def update_net(self, buffer) -> tuple:
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        conf = self.config
        constraints = weightConstraint()
        buffer.update_now_len()
        clip_norm = 1000
        obj_critic = obj_actor = None
        for update_c in range(
            int(1 + buffer.now_len * self.repeat_times / self.batch_size)
        ):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            # self.optimizer_update(self.cri_optimizer, obj_critic)
            self.optimizer_update_clip(self.cri_optimizer, obj_critic, self.cri, clip_norm)
            self.cri._modules['hidden_layer_1'].apply(constraints)
            self.cri._modules['hidden_layer_2'].apply(constraints)
            self.cri._modules['hidden_layer_q1'].apply(constraints)
            self.cri._modules['hidden_layer_q2'].apply(constraints)
            self.cri._modules['output_layer_q1'].apply(constraints)
            self.cri._modules['output_layer_q2'].apply(constraints)

            action_pg = self.act(state)  # policy gradient
            action_table = action_pg.clone()
            with torch.no_grad():
                for i in range(self.batch_size):
                    action_table[i] = projection_function(action_table[i].reshape(1,-1), state[i].reshape(1,-1),conf)
            loss = torch.nn.SmoothL1Loss()
            obj_actor = (self.cri_target(state, action_pg) + self.beta_1 * penalty_func(state, action_pg,conf) \
                        + self.beta_2 * loss(action_pg, action_table.detach())).mean()
            self.beta_1 = min(1e-4, self.beta_1 * (1+1e-3))
            self.beta_2 = max(100, self.beta_2 * (1-1e-3))
            # self.optimizer_update(self.act_optimizer, obj_actor)
            self.optimizer_update_clip(self.act_optimizer, obj_actor, self.act, clip_norm)
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
                self.cri_target._modules['hidden_layer_1'].apply(constraints)
                self.cri_target._modules['hidden_layer_2'].apply(constraints)
                self.cri_target._modules['hidden_layer_q1'].apply(constraints)
                self.cri_target._modules['hidden_layer_q2'].apply(constraints)
                self.cri_target._modules['output_layer_q1'].apply(constraints)
                self.cri_target._modules['output_layer_q2'].apply(constraints)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item() / 2, -obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action_noise(
                next_s, self.policy_noise
            )  # policy noise
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics
            q_label = reward + mask * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(
            q2, q_label
        )  # twin critics
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )
            next_a = self.act_target.get_action_noise(
                next_s, self.policy_noise
            )  # policy noise
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state
