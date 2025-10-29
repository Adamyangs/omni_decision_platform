# no longer use.
from tensorboardX import SummaryWriter

class Writer(object):

    def __init__(self):
        pass

    @classmethod
    def instance(cls, version, seed):
        if not hasattr(Writer, "_instance"):
            Writer._instance = SummaryWriter("tblogs/{}_{}/".format(version, seed))
        return Writer._instance 