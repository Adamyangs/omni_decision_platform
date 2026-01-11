from torch import nn    
from marl_models.Nets.multi_head_attention import MultiHeadAttentionProj

class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module=module
    
    def forward(self, input):
        return input + self.module(input)
    
    

class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        
        self.normalizer = nn.BatchNorm1d(embedding_dim, affine=True)
        
    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(input.size())


class MultiHeadAttentionLayer(nn.Sequential):
    r"""
    A layer with attention mechanism and normalization.
    
    for an embedding :math: `\pmb{x}`.
    
    ..math::
        \pmb{x} = \mathrm{MultiHeadAttentionLayer}(\pmb{x})
    
    The following is executed:
    
    .. math::
        \begin{aligned}
        \pmb{x}_0&=\pmb{x}+\mathrm{MultiHeadAttentionProj}(\pmb(X)) \\
        \pmb{x}_1&=\mathrm{BatchNorm}(\pmb{x}_0) \\
        \pmb{x}_2&=\pmb{x}_1+\mathrm{MLP_{\text{2 layers}}}(\pmb{x}_1) \\
        \pmb{h} &=\mathrm{BatchNorm}{\pmb{x}_2}
        \end{aligned}
        
    Args:
        n_heads: number of heads
        embedding_dim: dimension of the query, keys, values
        feed_forward_hidden: size of the hidden layer in the MLP
    Inputs: inputs
        `\pmb{x}`. [batch, graph_size, embedding_dim]
    Outputs: out
        `\pmb{h}` [batch, graph_size, embedding_dim]
    """
    
    def __init__(
        self,
        n_heads,
        embedding_dim,
        feed_forward_hidden=512
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttentionProj(
                    embedding_dim=embedding_dim,
                    n_heads=n_heads
                )
            ),
            Normalization(embedding_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embedding_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embedding_dim),
                )
                if feed_forward_hidden > 0
                else nn.Linear(embedding_dim, embedding_dim)
            ),
            Normalization(embedding_dim)
        )





class GraphAttentionEncoder(nn.Module):
    r"""
    Graph attention by self attention on graph nodes.
    
    For an embedding :math: '\pmb{x}', repeat 'n_layers' times:
    
    \pmb{h} = \mathrm{MultiHeadAttentionLayer}{\pmb{x}}0
    
    Args:
        n_heads: number of heads
        embedding_dim: dimension of the query, keys, values
        n_layers: number of : class: `~.MultiHeadAttentionLayer` to iterate.
        feed_forward_hidden: size of the hidden layer in the MLP
    Inputs: x
        embeddin: math: `\pmb{x}`. [batch, graph_size, embedding_dim]
    Outputs: (h, h_mean)
        the output `\pmb{h}` [batch, graph_size, embedding_dim]
    """
    
    def __init__(self, n_heads, embed_dim, n_layers, feed_forward_hidden=512):
        super(GraphAttentionEncoder, self).__init__()
        
        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
                for _ in range(n_layers)
            )
        )
        
    def forward(self, x):
        # [batch, cus_nodes+1(depot), embedding_dim]
        h = self.layers(x)
        # [batch, cus_nodes+1(depot), embedding_dim]
        return(h, h.mean(dim=1))
        