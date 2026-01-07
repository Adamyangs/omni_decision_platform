#### 测试流程

1. 将 Maps 文件夹下的地图放到游戏 `3rdparty/StarCraftII/Maps/SMAC_Maps` 目录下
2. 为了让 SMAC 支持新地图，需要将 smac_maps.py 文件替换 smac 的对应 python 文件，参考路径 `/python3.8/dist-packages/smac/env/starcraft2/maps/smac_maps.py`
3. 运行 `1_pretrain.py` 获取预训练模型
4. 运行 `2_multi_training.py` 进行 multi-task 训练
5. 在 results 目录下通过 tensorboard 获取胜率信息

#### 获取录像文件

1. 运行`3_eval.py`生成对应的 replay 文件,录像文件在`3rdparty/StarCraftII/Replays`下面
