# ============================================================
# new_DSO/trainer.py — 完整 DSO 训练器（第零步~第八步）
# ============================================================
import os
import json
import time
import signal
import logging
import warnings
from ND2.model import NDformer
import numpy as np
from socket import gethostname
from typing import List, Dict, Optional, Callable
from .config import get_default_config
from .vocabulary import Vocabulary
from .valid_mask import ValidMaskComputer
from .policy import TransformerPolicy
from .program import Program
from .gp_controller import NDGPController
from .risk_seeking import RiskSeekingSelector
from .baseline import BaselineComputer
from .policy_optimizer import DSOOptimizer
from ND2.GDExpr import GDExprClass, GDExpr
from ND2.utils import AttrDict, seed_all
import torch
from tqdm import tqdm
from ND2.search.reward_solver import RewardSolver
 
logger = logging.getLogger('new_DSO')
 
def _signal_handler(signum, frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
 
 
class NewDSOTrainer:
    """
    new_DSO 完整训练器，整合第零步到第八步。
 
    第八步职责:
    1. 每次迭代后更新全局最优
    2. 日志记录统计数据
    3. 早停判断（任务成功 / 采样数耗尽 / 时间到）
    4. 定期保存 checkpoint
    5. 训练结束后输出最终结果
    """
    def __init__(self,
                 config=None,
                 gdexpr_config=None,
                 Xv=None, Xe=None, A=None, G=None, Y=None, mask=None):
        
        available_vars = list(Xv.keys()) if Xv else []  # 如 ['v1', 'v2']
        if Xe:
            available_vars += list(Xe.keys())  # 如 ['e1']
        
        # ======== 第零步：初始化 ========
        self.config = config or get_default_config()
        seed_all(self.config.training.seed)
 
        self.vocab = Vocabulary(self.config)
        self.valid_mask_computer = ValidMaskComputer(self.vocab, self.config)
 
        self.policy = TransformerPolicy(self.vocab, self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)

        # ★ 加载预训练权重
        pretrain_path = './weights/new_dso_pretrained.pth'
        if os.path.exists(pretrain_path):
            ckpt = torch.load(pretrain_path, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(ckpt['policy'])
            logger.info(f"[new_DSO] 加载预训练权重: {pretrain_path} (epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?'):.4f})")
        else:
            logger.warning(f"[new_DSO] 预训练权重不存在: {pretrain_path}，使用随机初始化")
 
        self.policy.eval()
        for param in self.policy.parameters():
            param.requires_grad = False
        logger.info("[new_DSO] 策略网络已冻结，搜索过程中不更新参数")
        self.gdexpr = GDExprClass(self.config)
 
        # 数据
        self.Xv = Xv or {}
        self.Xe = Xe or {}
        self.A = A
        self.G = G
        self.Y = Y
        self.mask = mask

        self.ndformer = NDformer(device=self.device)
        self.ndformer.load('./weights/checkpoint.pth', weights_only=False)
        self.ndformer.eval()
        self.ndformer.set_data(
            Xv=Xv, Xe=Xe, A=A, G=G, Y=Y,
            root_type='node', cache_data_emb=True
        )       
 
        # 奖励计算器（第三步）
        if Y is not None and A is not None and G is not None:
            self.reward_solver = RewardSolver(
                Xv=Xv, Xe=Xe, A=A, G=G, Y=Y, mask=mask,
                complexity_base=self.config.data.complexity_base,
                sample_num=self.config.data.sample_num,
            )
        else:
            self.reward_solver = None
 
        # GP 控制器（第四步）
        gp_enabled = getattr(self.config, 'gp', None) is not None
        if gp_enabled and Y is not None:
            self.gp_controller = NDGPController(
                vocab=self.vocab, gdexpr=self.gdexpr, config=self.config,
                reward_solver=self.reward_solver,
                root_type=self.config.data.root_type,
            )
        else:
            self.gp_controller = None
 
        # 风险寻求选择器（第五步）
        tc = self.config.training
        self.risk_selector = RiskSeekingSelector(
            epsilon=tc.epsilon,
            use_memory_queue=True,
            memory_queue_size=10,
            memory_decay=0.9,
            min_elite=max(1, tc.batch_size // 20),
        )
 
        # 基线计算器（第六步）
        self.baseline_computer = BaselineComputer(
            mode=tc.baseline_mode,
            ewma_decay=0.1,
            combined_alpha=0.7,
        )
 
        # 策略优化器（第七步）
        self.optimizer = DSOOptimizer(
            policy=self.policy,
            vocab=self.vocab,
            valid_mask_computer=self.valid_mask_computer,
            device=self.device,
            entropy_weight=tc.entropy_weight,
            clip_grad_norm=100.0,
            learning_rate=tc.learning_rate,
            mode='PG',
            pqt_max_size=50,
            pqt_mix_ratio=0.2,
        )
 
        # ======== 第八步状态 ========
        self.n_evals = 0
        self.best_reward = -np.inf
        self.best_program = None
        self.best_metrics = {}
        self.best_prefix_with_coef = None
        self.start_time = None
        self.step_count = 0
        self.history = []  # 每步的摘要记录
 
        Program.clear_cache()
        Program.clear_reward_cache()
        logger.info("[new_DSO] 初始化完成（第零步~第八步就绪）")

        self.valid_mask_computer = ValidMaskComputer(
            self.vocab, self.config, available_vars=available_vars
        )
 
    # ============================================================
    # 第一步+第二步：采样 + 补全
    # ============================================================
    def sample_batch(self, batch_size=None):
        batch_size = batch_size or self.config.training.batch_size
        actions, log_probs, entropies, finished, masks = \
            self.policy.sample(
                batch_size=batch_size,
                valid_mask_computer=self.valid_mask_computer,
                device=self.device
                # prior_bias=self._get_prior_bias()
            )
        programs = []
        for i in range(batch_size):
            valid_mask_i = masks[i]
            token_ids = actions[i][valid_mask_i].cpu().tolist()
            cache_key = tuple(token_ids)
            if cache_key in Program._cache:
                programs.append(Program._cache[cache_key])
            else:
                prog = Program(
                    token_ids=token_ids, vocab=self.vocab,
                    config=self.config, gdexpr=self.gdexpr,
                    reward_solver=self.reward_solver,
                )
                Program._cache[cache_key] = prog
                programs.append(prog)
        self.n_evals += batch_size
        if self.step_count % 10 == 0:
            for i, p in enumerate(programs[:5]):  # 只打印前20个
                try:
                    expr_str = GDExpr.prefix2str(p.prefix)
                except:
                    expr_str = ' '.join(p.prefix)
                logger.info(f"[Sample] {i}: {expr_str} | terminal={p.is_terminal()}")
        return programs
 
    # ============================================================
    # 第三步：计算奖励
    # ============================================================
    def compute_rewards(self, programs):
        rewards = np.array([p.reward for p in programs])
        if rewards.max() > self.best_reward:
            for i, p in enumerate(programs):
                if rewards[i] > self.best_reward:
                    precise_reward = p.recompute_reward_precise()
                    rewards[i] = precise_reward
                    if precise_reward > self.best_reward:
                        self._update_best(p)
        # for i, p in enumerate(programs):
        #     has_var = any(self.vocab.kind(tid) in ('variable', 'node_coeff', 'edge_coeff') for tid in p.token_ids)
        #     if has_var:
        #         try:
        #             expr_str = GDExpr.prefix2str(p.prefix)
        #         except:
        #             expr_str = ' '.join(p.prefix)
        #         logger.info(f"[Reward] {i}: {expr_str} | reward={rewards[i]:.4f} | has_var=True")
        return rewards
 
    # ============================================================
    # 第四步：GP-Meld
    # ============================================================
    def gp_evolve(self, programs):
        if self.gp_controller is None:
            return []
        gp_programs = self.gp_controller.evolve(programs)
        if self.reward_solver is not None:
            for p in gp_programs:
                _ = p.reward
        return gp_programs
 
    # ============================================================
    # 第五步：风险寻求精英筛选
    # ============================================================
    def select_elite(self, programs, rewards):
        return self.risk_selector.select(programs, rewards)
    
 
    # ============================================================
    # 第六步：计算基线
    # ============================================================
    def compute_baseline(self, elite_rewards, all_rewards, quantile):
        return self.baseline_computer.compute(elite_rewards, all_rewards, quantile)
 
    # ============================================================
    # 第七步：策略梯度更新
    # ============================================================
    def policy_update(self, elite_programs, elite_rewards, baseline):
        # prior_bias = self._get_prior_bias()
        return self.optimizer.update(elite_programs, elite_rewards, baseline)
 
    # ============================================================
    # 第八步：记录与终止判断
    # ============================================================
    def _update_best(self, program):
        """更新全局最优"""
        if not program.is_terminal():
            return
        self.best_reward = program.reward
        self.best_program = program
        self.best_prefix_with_coef = program.prefix_with_coef
        if self.reward_solver is not None:
        # 先用 solve() 拟合系数，拿到 prefix_with_coef（<C> 已替换为数值）
            reward, prefix_with_coef = self.reward_solver.solve(
                program.prefix, sample=False
            )
            # 再用 evaluate() 计算各项指标
            self.best_metrics = self.reward_solver.evaluate(
                prefix_with_coef, {}
            )
            self.best_metrics['reward'] = reward
        logger.note(
            f"[new_DSO] ★ 新最优! reward={self.best_reward:.6f} | "
            f"R²={self.best_metrics.get('R2', 'N/A')} | "
            f"RMSE={self.best_metrics.get('RMSE', 'N/A')} | "
            f"expr={GDExpr.prefix2str(program.prefix_with_coef)}"
        )
    
    def _log_formulas(self, all_programs, all_rewards, elite_programs, elite_rewards,
                  step, file_path='./log/formulas.csv'):
        """
        将每步的中间公式写入文件。
        - Top-10 全部样本
        - 全部精英样本
        """
        import csv
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # ---- Top-10 全部样本 ----
            sorted_indices = np.argsort(all_rewards)[::-1]
            for rank, idx in enumerate(sorted_indices[:10]):
                p = all_programs[idx]
                r = all_rewards[idx]
                prefix_str = ' '.join(p.prefix)
                try:
                    # 优先用带系数的版本
                    if p.prefix_with_coef is not None:
                        expr_str = GDExpr.prefix2str(p.prefix_with_coef)
                    else:
                        expr_str = GDExpr.prefix2str(p.prefix)
                except Exception:
                    expr_str = prefix_str
                metrics = p.metrics if p.metrics else {}
                writer.writerow([
                    step, 'top10', rank,
                    f'{r:.6f}',
                    f"{metrics.get('R2', 'N/A')}",
                    f"{metrics.get('RMSE', 'N/A')}",
                    prefix_str, expr_str,
                ])
    
            # ---- 全部精英样本 ----
            for rank, (p, r) in enumerate(zip(elite_programs, elite_rewards)):
                prefix_str = ' '.join(p.prefix)
                try:
                    if p.prefix_with_coef is not None:
                        expr_str = GDExpr.prefix2str(p.prefix_with_coef)
                    else:
                        expr_str = GDExpr.prefix2str(p.prefix)
                except Exception:
                    expr_str = prefix_str
                metrics = p.metrics if p.metrics else {}
                writer.writerow([
                    step, 'elite', rank,
                    f'{r:.6f}',
                    f"{metrics.get('R2', 'N/A')}",
                    f"{metrics.get('RMSE', 'N/A')}",
                    prefix_str, expr_str,
                ])

    def _check_early_stop(self, early_stop_fn=None):
        """
        第八步早停判断:
        1. 若 early_stop_fn(best_metrics) → True → 停止
        2. 若总采样数 n_evals >= n_samples → 停止
        3. 若超时 → 停止
        """
        tc = self.config.training
        # 条件 1: 自定义早停
        if early_stop_fn is not None and early_stop_fn(self.best_metrics):
            logger.note(f"[new_DSO] 早停: 任务成功条件满足")
            return True
        # 条件 2: 采样数耗尽
        if self.n_evals >= tc.n_samples:
            logger.note(f"[new_DSO] 早停: 采样数耗尽 ({self.n_evals}/{tc.n_samples})")
            return True
        # 条件 3: 超时
        if hasattr(tc, 'time_limit') and tc.time_limit is not None:
            elapsed = time.time() - self.start_time
            if elapsed > tc.time_limit:
                logger.note(f"[new_DSO] 早停: 超时 ({elapsed:.0f}s > {tc.time_limit}s)")
                return True
        return False
 
    def _log_step(self, stats, elapsed):
        """第八步：日志记录"""
        log = {
            'Step': f'{self.step_count}',
            'Evals': f'{self.n_evals}',
            'Best-Reward': f'{self.best_reward:.4f}',
            'Best-R2': f"{self.best_metrics.get('R2', 'N/A')}",
            'Best-RMSE': f"{self.best_metrics.get('RMSE', 'N/A')}",
            'Best-Expr': GDExpr.prefix2str(self.best_prefix_with_coef)
                          if self.best_prefix_with_coef else 'None',
            'pg_loss': f"{stats.get('pg_loss', 'N/A')}",
            'Time': f'{elapsed:.1f}s',
        }
        logger.info(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))
 
    def _save_checkpoint(self, path):
        """保存训练状态"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'baseline': self.baseline_computer.state_dict(),
            'n_evals': self.n_evals,
            'best_reward': self.best_reward,
            'best_prefix_with_coef': self.best_prefix_with_coef,
            'best_metrics': self.best_metrics,
            'step_count': self.step_count,
            'config': dict(self.config),
        }
        torch.save(state, path)
        logger.info(f"[new_DSO] Checkpoint saved to {path}")
 
    def _load_checkpoint(self, path):
        """加载训练状态"""
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state['policy'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.baseline_computer.load_state_dict(state['baseline'])
        self.n_evals = state.get('n_evals', 0)
        self.best_reward = state.get('best_reward', -np.inf)
        self.best_prefix_with_coef = state.get('best_prefix_with_coef')
        self.best_metrics = state.get('best_metrics', {})
        self.step_count = state.get('step_count', 0)
        logger.info(f"[new_DSO] Checkpoint loaded from {path}")
 
    # ============================================================
    # ★ 完整训练循环（第零步~第八步）
    # ============================================================
    def _get_prior_bias(self):
        bias = {}
        # 鼓励图算子
        for token in ['aggr', 'sour', 'targ']:
            tid = self.vocab.word2id.get(token, -1)
            if tid >= 0:
                bias[tid] = 0.5
        # 抑制常数
        for token in ['1', '2', '3', '4', '5', '(1/2)', '(1/3)', '(1/4)', '(1/5)']:
            tid = self.vocab.word2id.get(token, -1)
            if tid >= 0:
                bias[tid] = -1.0
        # 变量稍微鼓励
        for token in ['v1', 'v2']:
            tid = self.vocab.word2id.get(token, -1)
            if tid >= 0:
                bias[tid] = 0.5
        # 抑制嵌套过深
        for token in ['exp', 'logabs', 'pow2', 'pow3', 'sigmoid', 'tanh']:
            tid = self.vocab.word2id.get(token, -1)
            if tid >= 0:
                bias[tid] = -0.5
        return bias

    def fit(self,
            early_stop_fn: Optional[Callable] = None,
            checkpoint_path: Optional[str] = None,
            log_every: int = 1,
            save_every: int = 100):
        """
        完整 DSO 训练循环。
 
        参数:
            early_stop_fn: Callable(dict) -> bool
                接收 best_metrics，返回 True 则停止。
                默认: R² > 0.99 或 ACC4 > 0.99
            checkpoint_path: str, 定期保存路径
            log_every: int, 每隔几步打印日志
            save_every: int, 每隔几步保存 checkpoint
        """
        formula_path = './log/new_dso/KUR/formulas.csv'
        os.makedirs(os.path.dirname(formula_path), exist_ok=True)
        with open(formula_path, 'w', encoding='utf-8') as f:
            f.write('step,category,rank,reward,R2,RMSE,prefix,expr\n')
        if early_stop_fn is None:
            early_stop_fn = lambda m: m.get('ACC4', 0) > 0.99 or m.get('R2', -np.inf) > 0.99
 
        self.start_time = time.time()
        tc = self.config.training
        logger.info(
            f"[new_DSO] 训练开始 | batch={tc.batch_size} | "
            f"ε={tc.epsilon} | n_samples={tc.n_samples} | "
            f"device={self.device}"
        )
        self.policy.eval()
 
        try:
            total_evals = 0   
            while total_evals < self.config.training.n_samples:
                step_start = time.time()
 
                # ---- 第一步+第二步: 采样 + 补全 ----
                programs = self.sample_batch()
 
                # ---- 第三步: 计算奖励 ----
                rewards = self.compute_rewards(programs)
 
                # ---- 第四步: GP-Meld ----
                gp_programs = self.gp_evolve(programs)
                gp_rewards = np.array([p.reward for p in gp_programs]) \
                             if gp_programs else np.array([])
 
                all_programs = programs + gp_programs
                # for i, p in enumerate(all_programs[:20]):
                #     try:
                #         if hasattr(p, 'prefix_with_coef') and p.prefix_with_coef is not None:
                #             expr_str = GDExpr.prefix2str(p.prefix_with_coef)
                #         else:
                #             expr_str = GDExpr.prefix2str(p.prefix)
                #     except:
                #         expr_str = ' '.join(p.prefix)
                #     logger.info(f"[Formula] {i}: reward={rewards[i] if i < len(rewards) else p.reward:.4f} | {expr_str}")
                all_rewards = np.concatenate([rewards, gp_rewards]) \
                              if len(gp_rewards) > 0 else rewards
 
                # ---- 第五步: 风险寻求精英筛选 ----
                elite_programs, elite_rewards, quantile = \
                    self.select_elite(all_programs, all_rewards)
                
                # ★ 写入中间公式
                self._log_formulas(
                    all_programs, all_rewards,
                    elite_programs, elite_rewards,
                    step=self.step_count,
                    file_path='./log/new_dso/KUR/formulas.csv',
                )
 
                # ---- 第六步: 计算基线 ----
                baseline = self.compute_baseline(elite_rewards, all_rewards, quantile)
 
                # ---- 第七步: 策略梯度更新 ----
                # n_updates = max(5, len(elite_programs) // 4)  # 至少5次
                n_updates = 3  # 固定更新次数，避免过拟合
                stats = {}
                for u in range(n_updates):
                    # stats = self.policy_update(elite_programs, elite_rewards, baseline)
                    if not stats or stats.get('total_loss', 1) == 0:
                        break
 
                # ---- 第八步: 记录与终止判断 ----
                elapsed = time.time() - step_start
 
                # 记录历史
                self.history.append({
                    'step': self.step_count,
                    'n_evals': self.n_evals,
                    'best_reward': self.best_reward,
                    'quantile': quantile,
                    'baseline': baseline,
                    'n_elite': len(elite_programs),
                    'n_gp': len(gp_programs),
                    'time': elapsed,
                    **stats,
                })
 
                # 日志
                if self.step_count % log_every == 0:
                    self._log_step(stats, time.time() - self.start_time)
 
                # 保存 checkpoint
                if checkpoint_path and self.step_count % save_every == 0:
                    self._save_checkpoint(checkpoint_path)
 
                # 早停判断
                if self._check_early_stop(early_stop_fn):
                    break

                total_evals += len(programs)
                self.n_evals += len(programs) 
                self.step_count += 1
                
 
        except KeyboardInterrupt:
            logger.info("[new_DSO] 手动中断")
        except Exception:
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self._finalize(checkpoint_path)
 
    def _finalize(self, checkpoint_path=None):
        """第八步：训练结束，输出最终结果"""
        total_time = time.time() - self.start_time if self.start_time else 0
        logger.note("=" * 60)
        logger.note("[new_DSO] 训练结束")
        logger.note(f"  总步数: {self.step_count}")
        logger.note(f"  总采样: {self.n_evals}")
        logger.note(f"  总耗时: {total_time:.1f}s")
        logger.note(f"  最优奖励: {self.best_reward:.6f}")
        if self.best_prefix_with_coef:
            # 改后
            try:
                expr_str = GDExpr.prefix2str(self.best_prefix_with_coef)
            except (ValueError, IndexError) as e:
                expr_str = ' '.join(str(x) for x in self.best_prefix_with_coef) + f"  (格式异常: {e})"
            logger.note(f"  最优表达式: {expr_str}")
        for k, v in self.best_metrics.items():
            logger.note(f"  {k}: {v}")
        logger.note("=" * 60)
 
        # 最终保存
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path)
 
    # ============================================================
    # 辅助
    # ============================================================