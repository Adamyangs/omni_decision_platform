REGISTRY = {}

from .rnn_agent import RNNAgent
from .commnet_agent import CommAgent
from .g2a_agent import G2AAgent
from .maven_agent import MAVENAgent
from .ant_agent import ANTAgent
from .ant_attention import ANTAttentionAgent
from .ant_demo import ANTDemoAgent
from .ant_demo_simu import ANTDemoSimuAgent
from .asn_agent import AsnRNNAgent
from .dyan_agent import DyAN
from .updet_agent import UpDetAgent
from .ant_multi import ANTMultiAgent




REGISTRY["rnn"] = RNNAgent
REGISTRY['commnet'] = CommAgent
REGISTRY['g2a'] = G2AAgent
REGISTRY['maven'] = MAVENAgent
REGISTRY['ant'] = ANTAgent
REGISTRY['ant_attention'] = ANTAttentionAgent
REGISTRY['ant_demo'] = ANTDemoAgent
REGISTRY['ant_demo_simu'] = ANTDemoSimuAgent
REGISTRY['asn'] = AsnRNNAgent
REGISTRY['dyan'] = DyAN
REGISTRY['updet'] = UpDetAgent
REGISTRY['ant_multi'] = ANTMultiAgent
