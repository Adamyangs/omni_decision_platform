FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Optional proxy settings (keep or override at build time)
ENV http_proxy=http://127.0.0.1:8889 \
    https_proxy=http://127.0.0.1:8889 \
    HTTP_PROXY=http://127.0.0.1:8889 \
    HTTPS_PROXY=http://127.0.0.1:8889 \
    no_proxy=localhost,127.0.0.1,::1 \
    NO_PROXY=localhost,127.0.0.1,::1

# CUDA paths and general env
ENV CUDA_PATH=/usr/local/cuda \
    CUDA_INCLUDE_PATH=/usr/local/cuda/include \
    CUDA_LIBRARY_PATH=/usr/local/cuda/lib64 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    SC2PATH=/root/StarCraftII

# Base OS deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    apt-utils software-properties-common ca-certificates \
    nano vim man build-essential wget curl git unzip sudo \
    libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev \
    libprotobuf-dev protobuf-compiler iproute2 lsof psmisc \
    python3 python3-pip python3-venv python3-dev gedit htop

# Python deps consolidated from scripts/install_dependencies.sh (no conda)
# gym>=0.24 fixes invalid requirement specifiers present in older releases
RUN pip3 install --upgrade "pip<24" setuptools wheel && \
    pip3 install \
      torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
        --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install \
      numpy==1.21.6 scipy==1.7.3 matplotlib==3.5.3 seaborn==0.12.2 \
      pyyaml==5.3.1 pygame==2.1.3 gym==0.24.1 pyglet==1.5.27 \
      pytest==7.4.4 probscale==0.2.3 imageio==2.31.6 snakeviz==2.2.0 \
      tensorboard==2.3.0 tensorboard-logger==0.1.0 jsonpickle==0.9.6 \
      protobuf==3.19.5 sacred==0.7.5 pymongo
RUN pip3 install git+https://github.com/oxwhirl/smac.git@26f4c4e4d1ebeaf42ecc2d0af32fac0774ccc678

# Install StarCraft II + SMAC maps inside the image
RUN mkdir -p /opt/install ${SC2PATH}/Maps && \
    cd /opt/install && \
    wget -q http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip && \
    unzip -q -P iagreetotheeula SC2.4.10.zip -d /root && \
    rm -f SC2.4.10.zip && \
    wget -q https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip && \
    unzip -q SMAC_Maps.zip && \
    mv SMAC_Maps ${SC2PATH}/Maps && \
    rm -f SMAC_Maps.zip

WORKDIR /workspace
