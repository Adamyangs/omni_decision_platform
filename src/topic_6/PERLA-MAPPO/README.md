# 指标5+6实验流程
## 环境配置
配置星际争霸环境
```sh
docker build -f docker/Dockerfile \
  --network=host --progress=plain \
  -t perla_image:demo .

# 运行容器（把当前仓库挂载到 /workspace/PERLA-MAPPO，容器默认工作目录在 /workspace）
docker run --name perla-train -itd --privileged --gpus all --network host \
  --entrypoint bash \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace/PERLA-MAPPO \
  perla_image:demo

docker exec -it perla-train bash
```
进入容器后SC2 与依赖已装好，无需再运行 install_sc2.sh 或 scripts/install_dependencies.sh
```sh
cp /workspace/smac/pymarl/maps/* ~/StarCraftII/Maps/SMAC_Maps/

chmod 777 -R ~/StarCraftII/Maps/SMAC_Maps/

cp /workspace/smac/pymarl/smac_maps.py /usr/local/lib/python3.8/dist-packages/smac/env/starcraft2/maps/smac_maps.py
```

## 指标5流程
```sh
cd /workspace/smac/pymarl
```
依次如下：
- 运行 `python3 1_pretrain.py` 获取预训练模型
- 运行 `python3 2_multi_training.py` 进行 multi-task 训练
在`tensorboard --logdir=/workspace/smac/results`命令下通过tensorboard获取胜率信息，然后切换到另一个Windows台式机看一下训好的策略的对局录像回放。

## 指标6流程
```sh
cd /workspace/PERLA-MAPPO
```
依次如下：
- 一键运行训练脚本`bash 1_run_training.sh`，相当运行PERLA_MAPPO以及两个baseline QSCAN和RODE在3s_vs_5z的训练（对应报告中的测试结果第一步）
- 因为时间过长，中途停止训练`bash 2_stop_training.sh`
- 运行`python3 3_plot_training_curve.py`，直接从之前训练好的`results`目录下提取tensorboard日志，并可视化训练对比结果，每一个方法perla/qscan/rode对应7个不同的tensorboard文件，会生成一个`training_results.pdf`文件（对应报告中的测试结果第二步）
- 运行`python3 4_plot_testing_win_rate.py`直接从之前存好的`models`目录下提取训好的actor模型，运行SC2后台对局看结果的胜率，有一定随机性但基本和原先报告里差不多，提升超过10%（对应报告中的测试结果第三步）


## 与指标无关的流程（只能在windows里面看可视化）
如果配合天泽，想看SC2游戏里的可视化，可以看到训练正常进行之后会有相应的ckpt文件，然后分别设置一下`checkpoint_path=$MODEL_DIR`、`runner='episide'`、`evaluate=True`以及`save_replay=True`。之后生成相应的replay文件在`/home/ps/pymarl/3rdparty/StarCraftII/Replays`目录，然后换成天泽那边的windows页面运行：
```sh
python3 -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```
已在启动流程里加了“端口可用性等待 + 快速失败重试”，避免 SC2 绑定失败时长时间卡住，再自动换端口继续尝试。
StarCraft2Env.py：改为 connect=False 启动 SC2，等待端口在 sc2_port_wait_timeout 内可连，再建立 websocket；失败就关进程并重试新端口，同时加了 Launching SC2 (attempt x/y) 日志。
新增可调参数（默认值）：sc2_port_wait_timeout=10.0、sc2_port_wait_interval=0.2、sc2_connect_timeout=10（秒）。
你可以直接重跑训练命令观察是否会自动重试并继续。如果还是卡住，建议先把等待/连接超时调小、重试次数调大，例如：
```sh
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=perla_mappo --env-config=sc2 \
  with env_args.map_name=3s_vs_5z env_args.sc2_port_retry_limit=10 \
  env_args.sc2_port_wait_timeout=5 env_args.sc2_connect_timeout=5
```