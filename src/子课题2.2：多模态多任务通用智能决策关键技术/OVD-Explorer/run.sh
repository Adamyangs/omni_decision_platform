# !/usr/bin/env bash

for ((seed=0; seed<5; seed+=1))
do
    
    version=10
    # DSAC
    # python main.py --seed=$seed --domain=ant --num_expl_steps_per_train_loop 1000 --num_trains_per_train_loop 1000 --use_aleatoric --version $version >logs/$version\_1$seed.log 2>&1 &
    # DOAC
    # python main.py --seed=$seed --domain=ant --num_expl_steps_per_train_loop 1000 --num_trains_per_train_loop 1000 --beta_UB=4.66 --delta=23.53 --use_aleatoric --version $version >logs/$version\_3$seed.log 2>&1 &
    version=13
    # OVDE_G
    # python main.py --seed=$seed --domain=ant --num_expl_steps_per_train_loop 1000 --num_trains_per_train_loop 1000 --alpha 0.05 --beta 3.2 --sigma 0 --z 0.5 --use_aleatoric --version $version --ee >logs/$version\_$seed.log 2>&1 &
    version=14
    # OVDE_Q
    # python main.py --seed=$seed --domain=ant --num_expl_steps_per_train_loop 1000 --num_trains_per_train_loop 1000 --alpha 0.05 --beta 3.2 --sigma 0 --z 0.5 --use_aleatoric --use_quantile_cdf --version $version --ee >logs/$version\_$seed.log 2>&1 &
    version=19
    # SAC
    # python main.py --seed=$seed --domain=ant --num_expl_steps_per_train_loop 1000 --num_trains_per_train_loop 1000 --version $version >logs/$version\_$seed.log 2>&1 &

    version=20
    # DSAC
    # python main.py --seed=$seed --domain=stochasticant --num_expl_steps_per_train_loop 100 --num_trains_per_train_loop 100 --use_aleatoric --version $version >logs/$version\_1$seed.log 2>&1 &
    # DoAC
    # python main.py --seed=$seed --domain=stochasticant --num_expl_steps_per_train_loop 100 --num_trains_per_train_loop 100 --beta_UB=4.66 --delta=23.53 --use_aleatoric --version $version >logs/$version\_3$seed.log 2>&1 &
    version=23
    # OVDE_G
    # python main.py --seed=$seed --domain=stochasticant --num_expl_steps_per_train_loop 100 --num_trains_per_train_loop 100 --alpha 0.05 --beta 3.2 --sigma 0 --z 0.5 --use_aleatoric --version $version --ee >logs/$version\_$seed.log 2>&1 &
    version=24
    # OVDE_Q
    # python main.py --seed=$seed --domain=stochasticant --num_expl_steps_per_train_loop 100 --num_trains_per_train_loop 100 --alpha 0.05 --beta 3.2 --sigma 0 --z 0.5 --use_aleatoric --use_quantile_cdf --version $version --ee >logs/$version\_$seed.log 2>&1 &
    version=29
    # SAC
    # python main.py --seed=$seed --domain=stochasticant --num_expl_steps_per_train_loop 100 --num_trains_per_train_loop 100 --version $version >logs/$version\_$seed.log 2>&1 &

done
