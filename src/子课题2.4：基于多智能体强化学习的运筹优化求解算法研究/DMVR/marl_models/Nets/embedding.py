"""
Problem specific node embedding for static feature.
"""

import torch
import torch.nn as nn

"""
AutoEmbedding
|-- VRPEmbedding
"""

def AutoEmbedding(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "marl_cvrp": MARL_VRPEmbedding
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding

class MARL_VRPEmbedding(nn.Module):
    """
    Embedding for the capacitated vehicle routing problem, solved by marl.
    The shape of tensors in ``input`` is summarized as following:

    +-----------+-------------------------+
    | key       | size of tensor          |
    +===========+=========================+
    | 'loc'     | [batch, n_customer, 2]  |
    +-----------+-------------------------+
    | 'depot'   | [batch, 2]              |
    +-----------+-------------------------+
    | 'demand'  | [batch, n_customer, 1]  |
    +-----------+-------------------------+

    Args:
        embedding_dim: dimension of output
    Inputs: input
        * **input** : dict of ['loc', 'depot', 'demand']
    Outputs: out
        * **out** : [batch, n_customer+1, embedding_dim]
    """

    def __init__(self, embedding_dim):
        super(MARL_VRPEmbedding, self).__init__()
        node_dim = 3  # x, y, demand

        self.context_dim = embedding_dim + 1  # Embedding of last node + remaining_capacity

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # depot embedding

    def forward(self, input):  # dict of 'loc', 'demand', 'depot'
        # env_num, 1, 2 -> batch, 1, embedding_dim
        depot_embedding = self.init_embed_depot(input["depot"])[:, None, :]
        # [env_num, n_customer, 2]  -> batch, n_customer, embedding_dim
        node_embeddings = self.init_embed(
            torch.cat((input["loc"], input["demand"][:, :, None]), -1)
        )
        # batch, n_customer+1, embedding_dim
        out = torch.cat((depot_embedding, node_embeddings), 1)
        return out