import numpy as np
import torch
from torch import nn

def load_from_list_of_dict(l_d:list, keys:list):
    """

    :param l_d: [{k1:v11, k2:v21}...{k1:v1n, k2:v2n}]
    :param keys: [k1, k2]
    :return: {k1:[v11,...,v1n], k2:[v21,...,v2n]}
    """
    items = {k:[d[k] for d in l_d] for k in keys}

    return items

def cal_switch_cost(prev_state, cur_state, cost):
    """

    :param prev_state:          np.array([n, ns])
    :param cur_state:           np.array([n, ns])
    :param cost:                np.array([ns, ns])
    :return:
    """
    assert (prev_state >= 0).all() and (prev_state <= 1).all() and (prev_state.sum(-1) == 1).all()
    assert (cur_state >= 0).all() and (cur_state <= 1).all() and (cur_state.sum(-1) == 1).all()

    to_other_costs = np.matmul(prev_state, cost)
    costs = (cur_state * to_other_costs).sum(-1)

    return costs

def to_onehot(idx, num_classes=None):
    # flatten
    shape = idx.shape
    idx = idx.flatten()

    # get max_id and num
    num_classes = idx.max() + 1 if num_classes is None else num_classes
    num = idx.shape[0]

    # idx to onehot
    code = np.zeros((num, num_classes))
    code[np.arange(num), idx.astype('int32')] = 1

    # unflatten
    code = np.reshape(code, [*shape, num_classes])

    return code

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

if __name__ == '__main__':
    debug = 0