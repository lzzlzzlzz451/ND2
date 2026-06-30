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
# from .policy import TransformerPolicy
from .policy import GNNTransformerPolicy  
from .program import Program
from .gp_controller import NDGPController
from .risk_seeking import RiskSeekingSelector
from .baseline import BaselineComputer
from .policy_optimizer import DSOOptimizer
from ND2.GDExpr import GDExprClass, GDExpr
from ND2.utils import AttrDict, seed_all
import torch
import torch.nn.functional as F
torch.cuda.set_device(2)
from tqdm import tqdm
from ND2.search.reward_solver import RewardSolver
 
logger = logging.getLogger('new_DSO')
 
def _signal_handler(signum, frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
 
 
class NewDSOTrainer:
    def __init__(self,
                 config=None,
                 gdexpr_config=None,
                 Xv=None, Xe=None, A=None, G=None, Y=None, mask=None):
 
        available_vars = list(Xv.keys()) if Xv else []
        if Xe:
            available_vars += list(Xe.keys())
 
        # ======== 第零步：初始化 ========
        self.config = config or get_default_config()
        seed_all(self.config.training.seed)
 
        self.vocab = Vocabulary(self.config)
        self.valid_mask_computer = ValidMaskComputer(self.vocab, self.config)
 
        # ★ 策略网络：GNN Encoder (512d) + 投影层 (512→128) + Transformer Decoder (128d)
        self.policy = GNNTransformerPolicy(self.vocab, self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
 
        # ★ 从 NDformer checkpoint 加载 GNN + encoder Transformer 权重（冻结）
        ndformer_path = './weights/checkpoint.pth'
        if os.path.exists(ndformer_path):
            self.policy.load_from_ndformer(ndformer_path, device=self.device)
            logger.info(f"[new_DSO] NDformer 编码器权重已加载: {ndformer_path}")
        else:
            logger.warning(f"[new_DSO] NDformer 权重不存在: {ndformer_path}，"
                           f"GNN 编码器将使用随机初始化（效果会很差）")
            
        self.root_type = getattr(self.config.data, 'root_type', 'node')

        # ★ 预训练 graph_proj
        if os.path.exists(ndformer_path) and Y is not None:
            self._pretrain_graph_proj(Xv, Xe, A, G, Y, self.root_type, mask, n_steps=200)

        # ★ 可选：加载旧版 new_DSO 预训练权重（解码器部分，strict=False 跳过不匹配的 key）
        pretrain_path = './weights/new_dso_pretrained.pth'
        if os.path.exists(pretrain_path):
            ckpt = torch.load(pretrain_path, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(ckpt['policy'], strict=False)
            logger.info(f"[new_DSO] 旧预训练权重已加载（解码器部分）: {pretrain_path}")
 
        self.gdexpr = GDExprClass(self.config)
 
        # 数据
        self.Xv = Xv or {}
        self.Xe = Xe or {}
        self.A = A
        self.G = G
        self.Y = Y
        self.mask = mask
 
        # ★ 预计算 graph_emb（二值化管线 → GNN + Transformer Encoder，只算一次）
        # self.root_type = getattr(self.config.data, 'root_type', 'node')
        self.graph_emb = None
        if Y is not None and A is not None and G is not None:
            self.graph_emb = self._encode_data(Xv, Xe, A, G, Y, self.root_type, mask)
            logger.info(f"[new_DSO] graph_emb shape: {self.graph_emb.shape}")
 
        # ★ 不再需要单独的 NDformer 实例，GNN 已在 policy 内部
        # （删掉原来的 self.ndformer = NDformer(...) 等代码）
 
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
 
        # 策略优化器（第七步）—— ★ 只优化 requires_grad=True 的参数
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
        # ★ 注入预计算的 graph_emb
        self.optimizer.set_graph_emb(self.graph_emb)
 
        # ======== 第八步状态 ========
        self.n_evals = 0
        self.best_reward = -np.inf
        self.best_program = None
        self.best_metrics = {}
        self.best_prefix_with_coef = None
        self.start_time = None
        self.step_count = 0
        self.history = []
 
        self._last_probs = None
        self._last_actions = None
        self._last_masks = None
        self._prob_vis_interval = 50
 
        Program.clear_cache()
        Program.clear_reward_cache()
 
        # 带 available_vars 的 mask computer（覆盖上面的）
        self.valid_mask_computer = ValidMaskComputer(
            self.vocab, self.config, available_vars=available_vars
        )
 
        # 日志
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        trainable_names = [n for n, p in self.policy.named_parameters() if p.requires_grad]
        frozen_params = [p for p in self.policy.parameters() if not p.requires_grad]
        logger.info(f"[new_DSO] 总参数: {sum(p.numel() for p in self.policy.parameters()):,}")
        logger.info(f"[new_DSO] 可训练参数: {sum(p.numel() for p in trainable_params):,}")
        logger.info(f"[new_DSO] 冻结参数: {sum(p.numel() for p in frozen_params):,}")
        logger.info(f"[new_DSO] 可训练模块: {set(n.split('.')[0] for n in trainable_names)}")
        logger.info(f"[new_DSO] 初始化完成（第零步~第八步就绪）")
 
    def _pretrain_graph_proj(self, Xv, Xe, A, G, Y, root_type, mask=None, n_steps=200):
        """
        预训练 graph_proj：用合成数据教投影层把 512d GNN 输出对齐到 128d 解码器空间。
 
        思路：用 NDformer 的解码器做"老师"，graph_proj 的输出要接近 NDformer 
        编码器的 512d 输出经过 NDformer 自己解码器能正确预测 token 的效果。
        简化方案：直接用 MSE 让 graph_proj(GNN_512d) ≈ GNN_512d[:, :128] 的 
        线性投影，但更好的方式是：
 
        实际方案：用 ND2 合成数据生成 (var_dict, prefix) 对，让解码器通过 
        graph_proj 输出的 graph_emb 预测下一个 token，做 MLE 预训练。
        """
        from ND2.GDExpr import GDExpr
        from ND2.dataset.generator import Generator
 
        logger.info(f"[pretrain_proj] 开始预训练 graph_proj，{n_steps} 步")
 
        # 冻结编码器，只训练 graph_proj + 解码器
        self.policy.freeze_encoder()
        for param in self.policy.graph_proj.parameters():
            param.requires_grad = True
        for param in self.policy.decoder.parameters():
            param.requires_grad = True
        for param in self.policy.token_embedding.parameters():
            param.requires_grad = True
        for param in self.policy.ln_f.parameters():
            param.requires_grad = True
        for param in self.policy.output_head.parameters():
            param.requires_grad = True
 
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=1e-3)
        generator = Generator()
 
        # 硬编码一些合成方程（与 pretrain.py 相同）
        equations = [
            (['add', 'v1', 'mul', '<C>', 'aggr', 'sin', 'sub', 'sour', 'v2', 'targ', 'v2'], 'node'),
            (['sub', 'sub', 'sub', 'mul', 'v2', 'pow3', 'v2', 'v1',
              'div', 'mul', '<C>', 'aggr', 'sub', 'sour', 'v2', 'targ', 'v2', 'aggr', '<C>'], 'node'),
            (['sub', 'add', '<C>', 'mul', '<C>', 'v2', 'mul', '<C>', 'v1'], 'node'),
            (['add', 'neg', 'v2', 'aggr', 'sour', 'regular', 'v2', '<C>'], 'node'),
            (['add', 'neg', 'v2', 'aggr', 'sour', 'sigmoid', 'mul', '<C>', 'sub', 'v2', '<C>'], 'node'),
            (['add', 'mul', 'neg', 'v3', 'v2',
              'aggr', 'mul', 'sub', '<C>', 'targ', 'v2', 'sour', 'v2'], 'node'),
        ]
 
        self.policy.train()
        for step in range(n_steps):
            # 随机选一个方程
            prefix, rt = equations[np.random.randint(len(equations))]
 
            # 生成合成数据
            try:
                var_dict = generator.generate_data(prefix, rt)
            except Exception:
                continue
 
            # 编码图数据
            A_arr = var_dict['A']
            G_arr = var_dict['G']
            out_arr = var_dict['out']
            N_gen, V_gen, E_gen = out_arr.shape[0], A_arr.shape[0], G_arr.shape[0]
 
            v = [out_arr if rt == 'node' else np.zeros((N_gen, V_gen))] + \
                [var_dict.get(var, np.zeros((N_gen, V_gen))) for var in GDExpr.variable.node]
            v = np.stack(v, axis=-1)
            e = [out_arr if rt == 'edge' else np.zeros((N_gen, E_gen))] + \
                [var_dict.get(var, np.zeros((N_gen, E_gen))) for var in GDExpr.variable.edge]
            e = np.stack(e, axis=-1)
 
            v_bits = torch.from_numpy(GDExpr.parse_float(v)).to(self.device, torch.float32)
            e_bits = torch.from_numpy(GDExpr.parse_float(e)).to(self.device, torch.float32)
            G_t = torch.from_numpy(G_arr).to(self.device, torch.long)
            A_t = torch.from_numpy(A_arr).to(self.device, torch.long)
 
            graph_emb = self.policy.encode_graph(v_bits, e_bits, G_t, A_t, rt)  # (N_s, 128)
            graph_emb = graph_emb.unsqueeze(0)  # (1, N_s, 128)
 
            # 构造 prefix token ids
            token_ids = [self.vocab.sos_id]
            for tok in prefix:
                mapped = {'term': 'targ'}.get(tok, tok)
                if mapped in self.vocab.word2id:
                    token_ids.append(self.vocab.word2id[mapped])
            token_ids.append(self.vocab.eos_id)
 
            if len(token_ids) < 3:
                continue
 
            prefix_t = torch.tensor([token_ids], dtype=torch.long, device=self.device)
 
            # 前向
            logits = self.policy.forward_full(prefix_t, graph_emb=graph_emb)  # (1, L, n_actions)
 
            # teacher forcing：每个位置预测下一个 token
            targets = prefix_t[:, 1:]  # (1, L-1)
            pred = logits[:, :-1, :]   # (1, L-1, n_actions)
 
            loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)),
                                   targets.reshape(-1),
                                   ignore_index=self.vocab.pad_id)
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
 
            if (step + 1) % 50 == 0:
                logger.info(f"[pretrain_proj] step {step+1}/{n_steps} loss={loss.item():.4f}")
 
        # 预训练完毕，重新冻结编码器
        if self.config.encoder.freeze:
            self.policy.freeze_encoder()
 
        logger.info(f"[pretrain_proj] 完成")
    
    # -------------------------------------------------------
    # ★ 图数据编码（与 core.py 中逻辑相同）
    # -------------------------------------------------------
    def _encode_data(self, Xv, Xe, A, G, Y, root_type, mask=None):
        self.policy.eval()
        with torch.no_grad():
            V, E, T = A.shape[0], G.shape[0], Y.shape[0]
    
            # 构建 var_dict（与之前一样）
            var_dict = dict(A=A, G=G, out=Y)
            for idx, (k, v) in enumerate(Xv.items(), 1):
                if v.ndim == 1: v = v.reshape(1, -1).repeat(T, axis=0)
                if v.shape[-1] == 1: v = v.repeat(V, axis=-1)
                var_dict[f'v{idx}'] = v
            for idx, (k, e) in enumerate(Xe.items(), 1):
                if e.ndim == 1: e = e.reshape(1, -1).repeat(T, axis=0)
                if e.shape[-1] == 1: e = e.repeat(E, axis=-1)
                var_dict[f'e{idx}'] = e
    
            out = var_dict['out']
            N = out.shape[0]
            v = [out if root_type == 'node' else np.zeros((N, V))] + \
                [var_dict.get(var, np.zeros((N, V))) for var in GDExpr.variable.node]
            v = np.stack(v, axis=-1)  # (N, V, 1+d_v)
            e = [out if root_type == 'edge' else np.zeros((N, E))] + \
                [var_dict.get(var, np.zeros((N, E))) for var in GDExpr.variable.edge]
            e = np.stack(e, axis=-1)  # (N, E, 1+d_e)
    
            # ★ 第一阶段：在时间维度上预采样（与 NDformer.encode 一致）
            ec = self.config.encoder
            VorE = (V if root_type == 'node' else E)
            n = int(np.ceil(3 * ec.max_sample_num / VorE))
            if n < N:
                sample_idx = np.random.choice(N, n, replace=False)
                v = v[sample_idx]
                e = e[sample_idx]
                N = n
                if mask is not None:
                    mask = mask[sample_idx]
                logger.info(f"[_encode_data] 时间步采样: {T} → {n}")
    
            # 第二阶段：二值化 → GNN → 展平截断（encode_graph 内部）
            v_bits = torch.from_numpy(GDExpr.parse_float(v)).to(self.device, torch.float32)
            e_bits = torch.from_numpy(GDExpr.parse_float(e)).to(self.device, torch.float32)
            G_t = torch.from_numpy(G).to(self.device, torch.long)
            A_t = torch.from_numpy(A).to(self.device, torch.long)
    
            graph_emb = self.policy.encode_graph(v_bits, e_bits, G_t, A_t, root_type, mask=mask)
            graph_emb = graph_emb.unsqueeze(0)
    
        return graph_emb
 
    # -------------------------------------------------------
    # 第一步+第二步：采样 + 补全（★ 传入 graph_emb）
    # -------------------------------------------------------
    def sample_batch(self, batch_size=None):
        batch_size = batch_size or self.config.training.batch_size
        actions, log_probs, entropies, finished, masks, probs = \
            self.policy.sample(
                batch_size=batch_size,
                valid_mask_computer=self.valid_mask_computer,
                device=self.device,
                graph_emb=self.graph_emb,    # ← 关键！
            )
        # 后续 Program 构建逻辑不变……
        programs = []
        for i in range(batch_size):
            valid_mask_i = masks[i]
            token_ids = actions[i][valid_mask_i].cpu().tolist()
            cache_key = tuple(token_ids)
            if cache_key in Program._cache:
                programs.append(Program._cache[cache_key])
            else:
                prog = Program(token_ids=token_ids, vocab=self.vocab, 
                                config=self.config, gdexpr=self.gdexpr, 
                                reward_solver=self.reward_solver)
                programs.append(prog)
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
            try:
                reward, prefix_with_coef = self.reward_solver.solve(
                    program.prefix, sample=False
                )
                self.best_metrics = self.reward_solver.evaluate(
                    prefix_with_coef, {}
                )
                self.best_metrics['reward'] = reward
            except Exception as e:
                logger.warning(f"[new_DSO] _update_best solve 失败: {e}，跳过该程序")
                return   # ← solve 失败就不更新 best
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
 
        try:
            total_evals = 0   
            while total_evals < self.config.training.n_samples:
                step_start = time.time()
 
                # ---- 第一步+第二步: 采样 + 补全 ----
                self.policy.eval()
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
                self.policy.train()
                n_updates = 3  # 固定更新次数，避免过拟合
                stats = {}
                for u in range(n_updates):
                    stats = self.policy_update(elite_programs, elite_rewards, baseline)
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

                if self.step_count % self._prob_vis_interval == 0 and self._last_probs is not None:
                    self._visualize_probs(self.step_count)
 
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
    def _visualize_probs(self, step):
        """从训练 batch 中选 top-3 表达式，可视化每步 token 概率分布"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    
        probs = self._last_probs          # (B, L, n_actions)
        actions = self._last_actions      # (B, L)
        masks = self._last_masks          # (B, L)
    
        token_names = [self.vocab.id2word.get(i, f'?{i}') 
                    for i in range(probs.shape[-1])]
    
        # 选序列最长的前3个样本（信息最丰富）
        lengths = masks.sum(dim=1)
        top_indices = torch.argsort(lengths, descending=True)[:3]
    
        save_dir = f'./log/prob_vis/step_{step:04d}'
        os.makedirs(save_dir, exist_ok=True)
    
        for sample_idx in top_indices.cpu().tolist():
            seq_len = int(lengths[sample_idx].item())
            if seq_len == 0:
                continue
    
            sampled_ids = actions[sample_idx, :seq_len].cpu().tolist()
            sampled_names = [token_names[tid] for tid in sampled_ids]
    
            # --- 每步 bar chart ---
            for t in range(seq_len):
                p = probs[sample_idx, t].cpu().numpy()    # (n_actions,)
                top_k = 15
                top_idx = np.argsort(p)[::-1][:top_k]
                names = [token_names[i] for i in top_idx]
                vals = [p[i] for i in top_idx]
                colors = ['#e74c3c' if i == sampled_ids[t] else '#3498db' 
                        for i in top_idx]
    
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(range(len(names)), vals, color=colors)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')
                ax.set_ylabel('Probability')
                ax.set_title(
                    f'Step {step} | Sample {sample_idx} | t={t} | '
                    f'sampled [{sampled_names[t]}] (p={p[sampled_ids[t]]:.4f})\n'
                    f'Expr: {" ".join(sampled_names[:t+1])}'
                )
                ax.set_ylim(0, max(vals) * 1.2 if vals else 1.0)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, 
                            bar.get_height() + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=7)
                plt.tight_layout()
                fig.savefig(f'{save_dir}/sample{sample_idx}_step{t:02d}.png', dpi=150)
                plt.close(fig)
    
            # --- 热力图：该样本所有步 × 高频 token ---
            all_top = set()
            for t in range(seq_len):
                p = probs[sample_idx, t].cpu().numpy()
                all_top.update(np.argsort(p)[::-1][:10].tolist())
            all_top = sorted(all_top)
    
            heatmap = np.array([probs[sample_idx, t, all_top].cpu().numpy() 
                            for t in range(seq_len)])
            xlabels = [token_names[i] for i in all_top]
            ylabels = [f't={t}' for t in range(seq_len)]
    
            fig, ax = plt.subplots(figsize=(max(8, len(all_top)*0.6), 
                                            max(3, seq_len*0.5)))
            im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd')
            ax.set_xticks(range(len(all_top)))
            ax.set_xticklabels(xlabels, rotation=45, ha='right')
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(ylabels)
            ax.set_title(f'Step {step} | Sample {sample_idx} | '
                        f'Expr: {" ".join(sampled_names)}')
            fig.colorbar(im, ax=ax, shrink=0.6)
            for t, tid in enumerate(sampled_ids):
                if tid in all_top:
                    x = all_top.index(tid)
                    ax.add_patch(plt.Rectangle((x-0.5, t-0.5), 1, 1,
                                fill=False, edgecolor='blue', lw=2))
            plt.tight_layout()
            fig.savefig(f'{save_dir}/sample{sample_idx}_heatmap.png', dpi=150)
            plt.close(fig)
    
        logger.info(f"[new_DSO] 概率可视化已保存到 {save_dir}/")