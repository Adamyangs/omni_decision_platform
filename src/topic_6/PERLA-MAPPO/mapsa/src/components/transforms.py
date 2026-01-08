import torch as th


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    # 说明: 将离散动作索引转换为 one-hot 表示, 供神经网络输入
    def __init__(self, out_dim):
        self.out_dim = out_dim

    # 说明: 根据张量最后一维索引生成 one-hot 张量
    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    # 说明: 返回转换后的形状与数据类型
    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32
