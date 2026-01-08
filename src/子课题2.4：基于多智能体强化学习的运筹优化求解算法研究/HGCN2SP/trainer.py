import os
import sys
import time
import torch
import numpy as np
import wandb
import pickle
import torch.nn as nn  
from agent import Agent
from env import CFLPEnv
from sample import Sampler
from utils import solve_cflp_softmax

class PPOTrainer:
    def __init__(self, args, policy_param, train_param, data, bs, clusters, run_name, device):
        self.args = args
        self.train_param = train_param
        self.run_name = run_name
        self.device = device
        self.envs = CFLPEnv(data, bs, clusters, train_param['sel_num'], args.num_envs, device=self.device)
        self.agent = Agent(policy_param, train_param, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
        self.sampler = Sampler(self.envs, self.agent, args.num_steps, self.device)
        self.best_eval_delta = 100.0
        self.eval_epoch = 0
        self.use_wandb = args.track and hasattr(wandb, 'run') and wandb.run is not None
        
        # 判断是否在终端中运行（nohup时会返回False）
        self.is_terminal = sys.stdout.isatty()
        
        # 导入tqdm并根据运行环境调整
        from tqdm import tqdm
        self.tqdm = tqdm
        
        # 设置日志级别
        self.log_level = args.log_level if hasattr(args, 'log_level') else 'INFO'
        
        # 初始化日志记录器
        self._init_logger()

    def _init_logger(self):
        """初始化日志系统"""
        import logging
        
        # 创建logger
        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(getattr(logging, self.log_level))
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 创建formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 控制台handler（仅在终端运行时显示）
            if self.is_terminal:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            
            # 文件handler（始终记录到文件）
            os.makedirs('logs', exist_ok=True)
            file_handler = logging.FileHandler(f'logs/{self.run_name}.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log(self, message, level='INFO'):
        """统一的日志记录方法"""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        
        # 同时打印到控制台（如果不在终端中，使用简单的print）
        if not self.is_terminal and level in ['INFO', 'WARNING', 'ERROR']:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {level} - {message}")

    def train(self):
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_done = torch.zeros(self.envs.batch_size, device=self.device)
        
        self.log(f"🚀 Starting training with {self.args.num_iterations} iterations", "INFO")
        
        # 创建iteration进度条（非终端环境下使用简单格式）
        if self.is_terminal:
            pbar_iter = self.tqdm(range(1, self.args.num_iterations + 1), 
                             desc="🚀 Training", 
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} iters [{elapsed}<{remaining}]",
                             colour='green')
        else:
            # 非终端环境下使用更简单的格式，避免控制字符
            pbar_iter = self.tqdm(range(1, self.args.num_iterations + 1), 
                             desc="Training", 
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} iters",
                             ascii=True,  # 使用ASCII字符，避免乱码
                             mininterval=5.0,  # 减少更新频率
                             maxinterval=10.0)

        start_time = time.time()
        
        for iteration in pbar_iter:
            if self.args.anneal_lr:
                self._anneal_learning_rate(iteration)

            # 数据采集
            next_obs, next_done = self.sampler.collect_trajectories(next_obs)

            # 计算优势与回报
            advantages, returns = self.sampler.compute_advantages_and_returns(self.args)

            # 获取批数据
            b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = self.sampler.get_batch_data(advantages, returns)

            # 策略优化
            avg_loss = self._update_policy(b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values)

            # 评估与保存模型
            if iteration % self.train_param['eval_epoch'] == 0:
                self._evaluate_and_save(iteration, pbar_iter)
            
            # 更新进度条信息
            pbar_iter.set_postfix({
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'loss': f'{avg_loss:.4f}'
            })
            
            # 定期记录日志
            if iteration % 10 == 0:
                self.log(f"Iteration {iteration}: loss={avg_loss:.4f}, lr={self.optimizer.param_groups[0]['lr']:.2e}")
        
        # 训练完成后保存模型 
        if self.args.save_model:
            model_path = f"runs/{self.run_name}/{self.args.exp_name}.pt"
            folder_path = os.path.dirname(model_path)
            os.makedirs(folder_path, exist_ok=True)
            torch.save(self.agent.state_dict(), model_path)
            self.log(f"💾 Model saved to {model_path}")
        
        # 关闭进度条
        pbar_iter.close()
        
        # 打印训练完成信息
        total_time = time.time() - start_time
        self.log("\n" + "="*60, "INFO")
        self.log("🎉 Training Completed!", "INFO")
        self.log(f"📊 Total iterations: {self.args.num_iterations}", "INFO")
        self.log(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.2f}min)", "INFO")
        self.log(f"📈 Best eval delta: {self.best_eval_delta:.2f}%", "INFO")
        self.log("="*60, "INFO")
        
        self.envs.close()
    
    def _update_policy(self, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values):
        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        
        # 计算总迭代次数
        total_updates = self.args.update_epochs * (self.args.batch_size // self.args.minibatch_size)
        if self.args.batch_size % self.args.minibatch_size != 0:
            total_updates += self.args.update_epochs
        
        # 创建策略更新进度条（非终端环境下简化）
        if self.is_terminal:
            pbar_update = self.tqdm(total=total_updates, 
                                  desc="🔄 Updating", 
                                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} updates",
                                  colour='cyan',
                                  leave=False)
        else:
            # 非终端环境下使用简单格式
            pbar_update = self.tqdm(total=total_updates, 
                                  desc="Updating", 
                                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                                  ascii=True,
                                  leave=True,  # 保留在文件中
                                  mininterval=2.0)
        
        avg_loss = 0
        update_count = 0
        last_log_time = time.time()
        
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                b_obs_m = [b_obs[i] for i in mb_inds]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs_m, b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()
                    clipfracs.append(clipfrac)

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                # 更新平均损失
                avg_loss += loss.item()
                update_count += 1
                
                # 更新进度条信息
                if self.is_terminal or time.time() - last_log_time > 2.0:
                    pbar_update.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'kl': f'{approx_kl.item():.4f}'
                    })
                    last_log_time = time.time()
                
                # 更新进度条
                pbar_update.update(1)

                if self.use_wandb:
                    wandb.log({
                        'update/loss': loss.item(),
                        'update/pg_loss': pg_loss.item(),
                        'update/v_loss': v_loss.item(),
                        'update/entropy': entropy_loss.item(),
                        'update/clipfrac': clipfrac,
                        'update/approx_kl': approx_kl.item()
                    })
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                self.log(f"🛑 Early stopping at epoch {epoch+1} (KL threshold reached)")
                break
        
        # 关闭策略更新进度条
        pbar_update.close()
        
        # 计算平均损失
        avg_loss = avg_loss / update_count if update_count > 0 else 0
        
        return avg_loss

    def _anneal_learning_rate(self, iteration):
        frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
        self.optimizer.param_groups[0]["lr"] = frac * self.args.learning_rate
    
    def eval_model(self, eval_cls_path, action, train_param):
        mean_bs = 0
        mean_agent = 0
        mean_time = 0
        delta = 0
        eval_num = min(100, len(eval_cls_path))
        
        self.log(f"📊 Evaluating {eval_num} samples")
        
        # 创建eval进度条
        if self.is_terminal:
            pbar_eval = self.tqdm(range(eval_num), 
                               desc="📊 Evaluating", 
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} samples",
                               colour='yellow',
                               leave=False)
        else:
            pbar_eval = self.tqdm(range(eval_num), 
                               desc="Evaluating", 
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                               ascii=True,
                               leave=True)
        
        results = []
        
        for i in pbar_eval:
            cls_loc = os.path.join(train_param["eval_cls_loc"], eval_cls_path[i]) 
            with open(cls_loc, 'rb') as f:
                cls = pickle.load(f)
            file_path = f"result_of_{eval_cls_path[i][10:-4]}.pkl"
            file_path = os.path.join(train_param["eval_result"], file_path)
            with open(file_path, "rb") as f:
                now_result = pickle.load(f)
                bs = now_result['primal']
                mean_bs += bs
            args = (cls, action[i].cpu(), True)
            eval_results = solve_cflp_softmax(args).squeeze()
            
            # 更新进度条信息
            if self.is_terminal:
                pbar_eval.set_postfix({
                    'agent': f'{eval_results[0].item():.2f}',
                    'gap': f'{(eval_results[0].item() - bs)/ bs *100:.2f}%'
                })
            
            gap = (eval_results[0].item() - bs)/ bs *100
            results.append({
                'sample': i+1,
                'agent': eval_results[0].item(),
                'bs': bs,
                'gap': gap,
                'time': eval_results[1].item()
            })
            
            mean_agent += eval_results[0].item()
            mean_time += eval_results[1].item()
            delta += gap

        # 关闭eval进度条
        pbar_eval.close()
        
        mean_agent = mean_agent / eval_num
        mean_bs = mean_bs / eval_num
        mean_time = mean_time / eval_num
        delta /= eval_num
        
        # 记录评估结果
        self.log("\n" + "="*60, "INFO")
        self.log("📊 EVALUATION SUMMARY", "INFO")
        self.log(f"  Average agent result: {mean_agent:.2f}", "INFO")
        self.log(f"  Average baseline: {mean_bs:.2f}", "INFO")
        self.log(f"  Average time: {mean_time:.2f}s", "INFO")
        self.log(f"  Average gap with Gurobi: {delta:.2f}%", "INFO")
        self.log("="*60, "INFO")
        
        # 保存详细结果
        os.makedirs('eval_results', exist_ok=True)
        result_file = f'eval_results/{self.run_name}_eval_{self.eval_epoch}.txt'
        with open(result_file, 'w') as f:
            f.write(f"Evaluation at epoch {self.eval_epoch}\n")
            f.write(f"Average agent: {mean_agent:.2f}\n")
            f.write(f"Average baseline: {mean_bs:.2f}\n")
            f.write(f"Average gap: {delta:.2f}%\n")
            f.write("\nDetailed results:\n")
            for r in results:
                f.write(f"Sample {r['sample']}: agent={r['agent']:.2f}, bs={r['bs']:.2f}, gap={r['gap']:.2f}%, time={r['time']:.2f}s\n")
        
        if delta < self.best_eval_delta:
            self.log(f"💾 Saving best model (new best delta: {delta:.2f}% < {self.best_eval_delta:.2f}%)")
            self.best_eval_delta = delta
            model_path = f"./model_path/{self.run_name}_eval_{self.eval_epoch}_{delta:.2f}.pt"
            torch.save(self.agent.state_dict(), model_path)
            self.log(f"✅ Model saved to {model_path}")
        
        self.eval_epoch += 1
        
        if self.use_wandb:
            wandb.log({
                "eval/mean_agent": mean_agent, 
                "eval/mean_bs": mean_bs, 
                "eval/mean_time": mean_time, 
                "eval/delta": delta,
                "eval/best_delta": self.best_eval_delta
            })
        
        return delta
    
    def _evaluate_and_save(self, iteration, pbar_iter=None):
        train_param = self.train_param
        
        # 保存当前模型
        model_path = f"./model_path/{self.run_name}_seed_{self.args.seed}_{iteration}.pt"
        torch.save(self.agent.state_dict(), model_path)
        
        # 在进度条中显示评估状态
        if pbar_iter:
            pbar_iter.set_description(f"Evaluating (Iter {iteration})")
        
        self.log(f"\n🔍 Starting evaluation at iteration {iteration}...")
        
        eval_data = torch.load(os.path.join(train_param['eval_path'], train_param['eval_pt']))   
        for i in range(len(eval_data)):
            eval_data[i] = eval_data[i].to(self.device)
            
        with open(os.path.join(train_param['eval_path'], train_param['eval_cls']), 'rb') as f:   
            eval_cls_path = pickle.load(f)
            
        with torch.no_grad():
            action, _, _, _ = self.agent.get_action_and_value(eval_data, decode_type="greedy")
            delta = self.eval_model(eval_cls_path, action, train_param)
        
        # 恢复训练进度条描述
        if pbar_iter:
            pbar_iter.set_description("Training")
            
        self.log(f"✅ Evaluation completed at iteration {iteration}, average gap: {delta:.2f}%\n")