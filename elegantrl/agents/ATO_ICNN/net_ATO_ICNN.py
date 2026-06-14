import torch
import torch.nn as nn
import torch.nn.functional as F


"""Actor (policy network)"""


class ActorICNN(nn.Module):
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


class CriticTwinICNN(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim, hidden_num = 3):
        super().__init__()
        self.input_layer = nn.Linear(state_dim + action_dim, mid_dim)
        self.hidden_layer_1 = nn.Linear(mid_dim, mid_dim, bias=False)
        self.hidden_layer_2 = nn.Linear(mid_dim, mid_dim, bias=False)
        self.hidden_layer_q1 = nn.Linear(mid_dim, mid_dim, bias=False)
        self.hidden_layer_q2 = nn.Linear(mid_dim, mid_dim, bias=False)
        self.output_layer_q1 = nn.Linear(mid_dim, 1, bias=False)
        self.output_layer_q2 = nn.Linear(mid_dim, 1, bias=False)
        self.passthrough_layer_1 = nn.Linear(state_dim + action_dim, mid_dim)
        self.passthrough_layer_2 = nn.Linear(state_dim + action_dim, mid_dim)
        self.passthrough_layer_3 = nn.Linear(state_dim + action_dim, mid_dim)
        self.passthrough_layer_4 = nn.Linear(state_dim + action_dim, mid_dim)
        self.passthrough_output_layer_q1 = nn.Linear(state_dim + action_dim, 1)
        self.passthrough_output_layer_q2 = nn.Linear(state_dim + action_dim, 1)

    def forward(self, state, action):
        return torch.add(*self.get_q1_q2(state, action)) / 2.0  # mean Q value

    def get_q_min(self, state, action):
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state, action):
        zx_1 = F.relu(self.input_layer(torch.cat((state, action), dim=1)))
        pass_1 = self.passthrough_layer_1(torch.cat((state, action), dim=1))
        zx_2 = F.relu(self.hidden_layer_1(zx_1) + pass_1)
        pass_2 = self.passthrough_layer_2(torch.cat((state, action), dim=1))
        zx_3 = F.relu(self.hidden_layer_2(zx_2) + pass_2)
        pass_3 = self.passthrough_layer_3(torch.cat((state, action), dim=1))
        zx_4_q1 = F.relu(self.hidden_layer_q1(zx_3) + pass_3)
        passout_q1 = self.passthrough_output_layer_q1(torch.cat((state, action), dim=1))
        zx_q1 = (self.output_layer_q1(zx_4_q1) + passout_q1)
        zx_4_q2 = F.relu(self.hidden_layer_q2(zx_3) + pass_3)
        passout_q2 = self.passthrough_output_layer_q2(torch.cat((state, action), dim=1))
        zx_q2 = (self.output_layer_q2(zx_4_q2) + passout_q2)
        return zx_q1, zx_q2  # two Q values