REGISTRY = {}

from .basic_controller import BasicMAC
from .maven_controller import MAVENMAC
from .ant_sim_controller import AntSimuMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["maven_mac"] = MAVENMAC
REGISTRY["ant_simu_controller"] = AntSimuMAC