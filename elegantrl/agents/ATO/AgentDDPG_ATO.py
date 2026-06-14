from elegantrl.agents.ATO.net_ATO import (
    Actor,
    CriticTwin,
)
from elegantrl.agents.ATO.AgentBase_ATO import AgentBase_ATO


class AgentDDPG_ATO(AgentBase_ATO):
    """
    Bases: ``AgentBase``

    Deep Deterministic Policy Gradient algorithm. “Continuous control with deep reinforcement learning”. T. Lillicrap et al.. 2015.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = True
        self.act_class = getattr(self, "act_class", Actor)
        self.cri_class = getattr(self, "cri_class", CriticTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.act.explore_noise = getattr(
            args, "explore_noise", 0.1
        )  # set for `get_action()`