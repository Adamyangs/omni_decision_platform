"""
StarCraft II Multi-Agent Reinforcement Learning Evaluation
Evaluates trained models on 3s_vs_5z scenario with 40 test episodes per algorithm
"""

import os
import sys
import time

# Import evaluation utilities
sys.path.append('./mapsa/src')
from utils.sc2_evaluator import evaluate_model

def main():
    """Main evaluation function"""
    print("=" * 60)
    print("StarCraft II MARL Algorithm Evaluation")
    print("Map: 3s_vs_5z | Episodes per algorithm: 50")
    print("=" * 60)
    
    # Model paths for evaluation
    models = [
        ("./models/perla_mappo_3s5z_best.pth", "PERLA_MAPPO"),
        ("./models/qscan_3s5z_best.pth", "QSCAN"), 
        ("./models/rode_3s5z_best.pth", "RODE")
    ]
    
    results_summary = []
    
    for model_path, algorithm in models:
        print(f"\n{'='*20} {algorithm} 评估 {'='*20}")
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            continue
        
        # Run evaluation
        results, win_rate = evaluate_model(model_path, "3s_vs_5z", 50)
        
        if results is not None:
            wins = sum(results)
            losses = 50 - wins
            results_summary.append((algorithm, wins, losses, win_rate))
        
        print(f"\n{'-'*50}")
        time.sleep(1)  # Brief pause between evaluations
    
    # Print final summary
    print(f"\n{'='*60}")
    print("评估结果汇总:")
    print(f"{'='*60}")
    for algorithm, wins, losses, win_rate in results_summary:
        print(f"{algorithm:12} | 胜: {wins:2d} 场 | 负: {losses:2d} 场 | 胜率: {win_rate:.3f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()