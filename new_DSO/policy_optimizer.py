# ============================================================
# new_DSO/policy_optimizer.py — 策略梯度优化器
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Optional, Tuple
from .program import Program
from .vocabulary import Vocabulary
from .policy import TransformerPolicy
from .valid_mask import ValidMaskComputer
 
logger = logging.getLogger('new_DSO.PolicyOptimizer')
 
 
class PQTBuffer:
    """
    Prioritized Queue Training (PQT) 缓冲区。
    维护历史最佳样本，训练时混入 elite batch。
    """
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._buffer = []  # List of (reward, token_ids, prefix)
 
    def push(self, programs: List[Program], rewards: np.ndarray):
        for p, r in zip(programs, rewards):
            if not np.isfinite(r) or not p.is_terminal():
                continue
            self._buffer.append((r, tuple(p.token_ids), p.prefix))
        # 只保留 top-k
        self._buffer.sort(key=lambda x: x[0], reverse=True)
        self._buffer = self._buffer[:self.max_size]
 
    def sample(self, n: int) -> List[Tuple[Tuple[int, ...], List[str]]]:
        if not self._buffer:
            return []
        n = min(n, len(self._buffer))
        indices = np.random.choice(len(self._buffer), n, replace=False)
        return [(self._buffer[i][1], self._buffer[i][2]) for i in indices]
 
    def __len__(self):
        return len(self._buffer)
 
 
class DSOOptimizer:
    """
    第七步：策略梯度更新。
 
    核心: 对 elite 样本重新前向传播 RNN，计算 neglogp，然后:
        pg_loss    = mean((r - baseline) * neglogp)
        entropy_loss = -entropy_weight * mean(entropy)
        loss = pg_loss + entropy_loss
 
    关键细节:
    - 采样阶段 self.policy.sample() 在 no_grad 下完成
    - 此处需要对 elite 的 action 序列重新跑一遍 RNN forward，这次带梯度
    - 逐步输入 action → 得到 logits → softmax → 计算交叉熵 → 得到 neglogp
    - 用 valid_mask 保证只对合法位置计算 loss
 
    支持 PG / PQT 两种模式:
    - PG (Vanilla Policy Gradient): 默认，只用当前 elite
    - PQT: 额外从历史优先队列中采样混入训练
    """
    def __init__(self,
                 policy: TransformerPolicy,
                 vocab: Vocabulary,
                 valid_mask_computer: ValidMaskComputer,
                 device: str = 'cpu',
                 entropy_weight: float = 0.01,
                 clip_grad_norm: float = 100.0,
                 learning_rate: float = 1e-3,
                 mode: str = 'PG',
                 pqt_max_size: int = 50,
                 pqt_mix_ratio: float = 0.2):
        self.policy = policy
        self.vocab = vocab
        self.valid_mask_computer = valid_mask_computer
        self.device = device
        self.entropy_weight = entropy_weight
        self.clip_grad_norm = clip_grad_norm
        self.mode = mode  # 'PG' or 'PQT'
 
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )
 
        # PQT 缓冲区
        self.pqt_buffer = PQTBuffer(max_size=pqt_max_size)
        self.pqt_mix_ratio = pqt_mix_ratio
 
        # 训练统计
        self._step_count = 0
 
    def update(self, elite_programs, elite_rewards, baseline, prior_bias=None):
        if not elite_programs: return {}
        self.policy.train()
        advantages = torch.tensor(elite_rewards - baseline, dtype=torch.float32, device=self.device)
        pg_loss, entropy_loss = self._compute_losses(elite_programs, advantages, prior_bias)
        if self.mode == 'PQT' and len(self.pqt_buffer) > 0:
            pql = self._pqt_step(prior_bias)
            if pql is not None: pg_loss = pg_loss + 0.5 * pql
        total_loss = pg_loss + entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
    
        # ★ NaN 梯度保护：检测到 NaN 就跳过更新
        has_nan = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in self.policy.parameters()
        )
        if has_nan:
            logger.warning("[Optimizer] NaN 梯度，跳过更新")
            self.optimizer.zero_grad()
            return {'pg_loss': 0.0, 'entropy_loss': 0.0, 'total_loss': 0.0,
                    'grad_norm': 0.0, 'n_elite': len(elite_programs), 'baseline': baseline}
    
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self._step_count += 1
        if self.mode == 'PQT': self.pqt_buffer.push(elite_programs, elite_rewards)
        stats = {'pg_loss': pg_loss.item(), 'entropy_loss': entropy_loss.item(), 'total_loss': total_loss.item(),
                'grad_norm': gn.item(), 'advantage_mean': advantages.mean().item(), 'n_elite': len(elite_programs), 'baseline': baseline}
        logger.info(f"[Optimizer] pg={stats['pg_loss']:.4f} ent={stats['entropy_loss']:.4f} total={stats['total_loss']:.4f} grad={stats['grad_norm']:.2f}")
        return stats
 
    def _compute_losses(self, programs, advantages, prior_bias):
        max_len = max(len(p.token_ids) for p in programs)
        E = len(programs)
 
        # 构造输入序列：SOS + token_ids + padding
        actions_t = torch.full((E, max_len + 1), self.vocab.pad_id,
                               dtype=torch.long, device=self.device)
        lengths = []
        for i, p in enumerate(programs):
            L = len(p.token_ids)
            # 位置 0 = SOS，位置 1~L = token_ids
            actions_t[i, 0] = self.vocab.sos_id
            actions_t[i, 1:L+1] = torch.tensor(p.token_ids, dtype=torch.long)
            lengths.append(L)
 
        # ★ 一次性前向传播整个序列
        all_logits = self.policy.forward_full(actions_t)  # (E, L+1, n_actions)
 
        # 逐步计算 nll 和 entropy（和 mask 相关逻辑不变）
        all_nll = []
        all_ent = []
        all_w = []
 
        has_var = torch.zeros(E, dtype=torch.bool, device=self.device)
        danglings = torch.ones(E, dtype=torch.long)
        coeff_c = torch.zeros(E, dtype=torch.long)
        coeff_cv = torch.zeros(E, dtype=torch.long)
        coeff_ce = torch.zeros(E, dtype=torch.long)
        finished = torch.zeros(E, dtype=torch.bool)
 
        for t in range(max_len):
            # logits for position t+1 (预测第 t 个 token)
            logits = all_logits[:, t, :]  # (E, n_actions)
 
            if prior_bias:
                for tid, b in prior_bias.items():
                    if tid >= 0: logits[:, tid] += b
 
            # 计算 valid mask（和原来一样）
            pl, dl, cl = [], [], []
            for i in range(E):
                if finished[i]:
                    pl.append([]); dl.append(0); cl.append((0, 0, 0))
                else:
                    va = actions_t[i, 1:t+1].cpu().tolist()
                    va = [a for a in va if a != self.vocab.pad_id]
                    pl.append(va)
                    dl.append(danglings[i].item())
                    cl.append((coeff_c[i].item(), coeff_cv[i].item(), coeff_ce[i].item()))
            has_var_list = has_var.cpu().tolist()
            vm = self.valid_mask_computer.compute_mask_batch(pl, dl, cl, has_variables=has_var_list)
            vmt = torch.from_numpy(vm).to(self.device)
            for i in range(E): vmt[i, :] = vmt[i, :] & ~finished[i]
 
            logits = logits.masked_fill(~vmt, -1e8)
            all_masked = (logits <= -1e7).all(dim=-1)
            logits[all_masked, 0] = 0.0
 
            log_p = F.log_softmax(logits, dim=-1)
            prob = torch.exp(log_p)
            ent = -(prob * log_p).sum(dim=-1)
            ent = torch.nan_to_num(ent, nan=0.0)
 
            # 目标 action
            act = actions_t[:, t + 1]  # 第 t 个 token
            nll = F.nll_loss(log_p, act, reduction='none', ignore_index=self.vocab.pad_id)
            nll = torch.nan_to_num(nll, nan=0.0)
 
            vp = torch.tensor([t < lengths[i] and not finished[i] for i in range(E)],
                              dtype=torch.bool, device=self.device)
            all_nll.append(nll)
            all_ent.append(ent)
            all_w.append(vp.float())
 
            # 更新状态追踪
            for i in range(E):
                if finished[i]: continue
                tid = act[i].item()
                if tid == self.vocab.pad_id:
                    finished[i] = True; continue
                danglings[i] = danglings[i] - 1 + self.vocab.arity(tid)
                k = self.vocab.kind(tid)
                if k == 'coefficient': coeff_c[i] += 1
                elif k == 'node_coeff': coeff_cv[i] += 1
                elif k == 'edge_coeff': coeff_ce[i] += 1
                if k == 'variable': has_var[i] = True
                if danglings[i] <= 0: finished[i] = True
 
        nll_m = torch.stack(all_nll, 1)
        ent_m = torch.stack(all_ent, 1)
        w_m = torch.stack(all_w, 1)
        adv_exp = advantages.unsqueeze(1).expand_as(nll_m)
        tw = w_m.sum()
        if tw > 0:
            pg = (adv_exp * nll_m * w_m).sum() / tw
            ent_mean = (ent_m * w_m).sum() / tw
        else:
            pg = torch.tensor(0.0, device=self.device)
            ent_mean = torch.tensor(0.0, device=self.device)
        ent_loss = -self.entropy_weight * ent_mean
        pg = pg / (pg.detach().abs().mean() + 1e-8)     # 归一化到 ~1.0
        ent_mean = ent_mean / (ent_mean.detach().abs().mean() + 1e-8)  # 归一化到 ~1.0
        ent_loss = -self.entropy_weight * ent_mean
        return pg, ent_loss
    
    def _pqt_step(self, prior_bias=None) -> Optional[torch.Tensor]:
        """
        PQT 模式：从历史优先队列中采样，用 MLE 损失训练。
        即最大化历史最优样本的 log 概率。
        """
        n_pqt = max(1, int(self.pqt_mix_ratio * len(self.pqt_buffer)))
        samples = self.pqt_buffer.sample(n_pqt)
        if not samples:
            return None
 
        # 构造伪 Program 用于前向传播
        pseudo_programs = []
        for token_ids_tuple, prefix in samples:
            prog = Program.__new__(Program)
            prog.token_ids = list(token_ids_tuple)
            prog.prefix = prefix
            prog.vocab = self.vocab
            prog.config = self.policy.config
            prog._reward = None
            pseudo_programs.append(prog)
 
        # PQT 用均匀 advantage = 1.0（即最大化 log 概率）
        advantages = torch.ones(len(pseudo_programs), device=self.device)
        pg_loss, _ = self._compute_losses(pseudo_programs, advantages, prior_bias)
        return pg_loss
 
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'step_count': self._step_count,
            'pqt_buffer': self.pqt_buffer._buffer,
        }
 
    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self._step_count = state.get('step_count', 0)
        if 'pqt_buffer' in state:
            self.pqt_buffer._buffer = state['pqt_buffer']