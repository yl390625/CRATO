import numpy as np
import torch

def projection_function(action_ts, state_ts, config):
    conf = config
    num_vars = conf.NoComponents + conf.NoProducts
    state = state_ts.detach().cpu().numpy()
    action = action_ts.detach().cpu().numpy()
    state = state.flatten()*(conf.observation_high-conf.observation_low)+conf.observation_low
    action = action.flatten()
    action = (action + 1) * conf.action_high / 2

    # Prepare parameters
    index = [sum(conf.L[:i + 1]) - 1 for i in range(conf.NoComponents)]
    IO = state[sum(conf.L):-conf.NoProducts] + state[index]
    BO = state[-conf.NoProducts:]

    x = action
    action[action < 0] = 0
    IO[IO <= 1e-3] = 0
    BO[BO <= 1e-3] = 0

    # Scale of production
    P = np.array(x[conf.NoComponents:] + 10e-15)
    P = np.minimum(BO, P) + 10e-15
    inventory_arg_zero = np.argwhere(IO == 0)
    for i in range(conf.NoProducts):
        if sum(conf.usage_rate[inventory_arg_zero, i]) != 0:
            P[i] = 0 + 10e-15

    inventory_constr_scale = (IO / np.array(np.dot(P, conf.usage_rate.T))).flatten()
    constr_scale = inventory_constr_scale
    constr_scale = constr_scale[constr_scale != 0]
    action[conf.NoComponents:] = P

    if constr_scale.size != 0:
        k = constr_scale.clip(0, 1).min()
        action[conf.NoComponents:] = k * P
    else:
        action[conf.NoComponents:] = 0

    action = np.array([2 * action[i] / conf.action_high[i] - 1 for i in range(num_vars)])

    return torch.tensor(action).reshape(1, -1)


def penalty_func(state, action_pg, config=None):
    conf = config
    observation_high = torch.tensor(conf.observation_high, device='cuda:0', dtype=torch.float32)
    observation_low = torch.tensor(conf.observation_low, device='cuda:0', dtype=torch.float32)
    state = state*(observation_high-observation_low)+observation_low
    index = [sum(conf.L[:i + 1]) - 1 for i in range(conf.NoComponents)]
    inventory_constr = state[:, sum(conf.L):-conf.NoProducts] + state[:, index]
    backorder_constr = state[:, -conf.NoProducts:]
    action_high = torch.tensor(conf.action_high, device='cuda:0', dtype=torch.float32)

    infeasible_inventory = (inventory_constr - torch.mm(
        (action_pg[:, conf.NoComponents:] + 1) * action_high[conf.NoComponents:] / 2,
        torch.tensor(conf.usage_rate.T, device='cuda:0', dtype=torch.float32)))
    infeasible_backorder = backorder_constr - (action_pg[:, conf.NoComponents:] + 1) * action_high[
                                                                                       conf.NoComponents:] / 2
    barrier = (- 1e-2 * torch.nan_to_num(torch.log(infeasible_inventory.clip(0) + 1e-3))).sum(axis=1) + \
              (- torch.nan_to_num(torch.log(infeasible_backorder.clip(0) + 1e-3))).sum(axis=1)
    return barrier