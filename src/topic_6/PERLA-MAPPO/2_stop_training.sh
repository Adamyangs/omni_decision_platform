#!/bin/bash

pkill -9 -u "$(whoami)" -f "src/main.py"
pkill -9 -u "$(whoami)" -f "ant_metaq_v5"
pkill -9 -u "$(whoami)" -f "2_multi_training.py"
pkill -9 -u "$(whoami)" -f "1_pretrain.py"
pkill -9 -u "$(whoami)" python
pkill -9 Main_Thread

# kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')

if [ "$1" = "all" ]; then
    rm -rf results/sacred
fi
