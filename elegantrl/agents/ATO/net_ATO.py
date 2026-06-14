import torch
import torch.nn as nn

"""Actor (policy network)"""


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )
        self.explore_noise = 0.1  # standard deviation of exploration action noise

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):  # for exploration
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * self.explore_noise).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

    def get_action_noise(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


"""Critic (value network)"""


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_sa = nn.Sequential(
            nn.Linear(state_dim + action_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )  # concat(state, action)
        self.net_q1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, 1)
        )  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, 1)
        )  # q2 value

    def forward(self, state, action):
        return torch.add(*self.get_q1_q2(state, action)) / 2.0  # mean Q value

    def get_q_min(self, state, action):
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values
