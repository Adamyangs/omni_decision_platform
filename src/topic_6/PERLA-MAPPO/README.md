# 指标6实验流程
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

## 指标6流程
```sh
cd /workspace/PERLA-MAPPO
```
依次如下：
- 一键运行训练脚本`bash 1_run_training.sh`，相当运行PERLA_MAPPO以及两个baseline QSCAN和RODE在3s_vs_5z的训练（对应报告中的测试结果第一步）
- 因为时间过长，中途停止训练`bash 2_stop_training.sh`
- 运行`python3 3_plot_training_curve.py`，直接从之前训练好的`results`目录下提取tensorboard日志，并可视化训练对比结果，每一个方法perla/qscan/rode对应7个不同的tensorboard文件，会生成一个`training_results.pdf`文件（对应报告中的测试结果第二步）
- 运行`python3 4_plot_testing_win_rate.py`直接从之前存好的`models`目录下提取训好的actor模型，运行SC2后台对局看结果的胜率，有一定随机性但基本和原先报告里差不多，提升超过10%（对应报告中的测试结果第三步）
