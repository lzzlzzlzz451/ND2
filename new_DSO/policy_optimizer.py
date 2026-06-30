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
from .policy import GNNTransformerPolicy
from .valid_mask import ValidMaskComputer
 
logger = logging.getLogger('new_DSO.PolicyOptimizer')
 
 
class PQTBuffer:
    """Prioritized Queue Training (PQT) 缓冲区。"""
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._buffer = []
 
    def push(self, programs: List[Program], rewards: np.ndarray):
        for p, r in zip(programs, rewards):
            if not np.isfinite(r) or not p.is_terminal():
                continue
            self._buffer.append((r, tuple(p.token_ids), p.prefix))
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
 
    ★ 改造：forward_full 调用传入 graph_emb；优化器只更新 requires_grad 参数。
    """
    def __init__(self,
                 policy: GNNTransformerPolicy,
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
        self.mode = mode
 
        # ★ 只优化 requires_grad=True 的参数（冻结的编码器不参与）
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        logger.info(f"[Optimizer] 可训练参数: {sum(p.numel() for p in trainable_params):,} / "
                     f"{sum(p.numel() for p in self.policy.parameters()):,}")
 
        # PQT 缓冲区
        self.pqt_buffer = PQTBuffer(max_size=pqt_max_size)
        self.pqt_mix_ratio = pqt_mix_ratio
 
        # 训练统计
        self._step_count = 0
 
        # ★ graph_emb 引用（由外部设置）
        self.graph_emb = None
 
    def set_graph_emb(self, graph_emb):
        """设置预计算的图 embedding（由 NewDSO.encode_data 产出）"""
        self.graph_emb = graph_emb
 
    def update(self, elite_programs, elite_rewards, baseline, prior_bias=None):
        if not elite_programs:
            return {}
        self.policy.train()
        advantages = torch.tensor(elite_rewards - baseline, dtype=torch.float32, device=self.device)
        pg_loss, entropy_loss = self._compute_losses(elite_programs, advantages, prior_bias)
        if self.mode == 'PQT' and len(self.pqt_buffer) > 0:
            pql = self._pqt_step(prior_bias)
            if pql is not None:
                pg_loss = pg_loss + 0.5 * pql
        total_loss = pg_loss + entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
 
        # NaN 梯度保护
        has_nan = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in self.policy.parameters() if p.requires_grad
        )
        if has_nan:
            logger.warning("[Optimizer] NaN 梯度，跳过更新")
            self.optimizer.zero_grad()
            return {'pg_loss': 0.0, 'entropy_loss': 0.0, 'total_loss': 0.0,
                    'grad_norm': 0.0, 'n_elite': len(elite_programs), 'baseline': baseline}
 
        # ★ 只 clip 可训练参数的梯度
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        gn = torch.nn.utils.clip_grad_norm_(trainable_params, self.clip_grad_norm)
        self.optimizer.step()
        self._step_count += 1
        if self.mode == 'PQT':
            self.pqt_buffer.push(elite_programs, elite_rewards)
        stats = {
            'pg_loss': pg_loss.item(), 'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(), 'grad_norm': gn.item(),
            'advantage_mean': advantages.mean().item(),
            'n_elite': len(elite_programs), 'baseline': baseline
        }
        logger.info(f"[Optimizer] pg={stats['pg_loss']:.4f} ent={stats['entropy_loss']:.4f} "
                     f"total={stats['total_loss']:.4f} grad={stats['grad_norm']:.2f}")
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
            actions_t[i, 0] = self.vocab.sos_id
            actions_t[i, 1:L+1] = torch.tensor(p.token_ids, dtype=torch.long)
            lengths.append(L)
 
        # ★ 传入 graph_emb
        all_logits = self.policy.forward_full(actions_t, graph_emb=self.graph_emb)
 
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
            logits = all_logits[:, t, :]
 
            if prior_bias:
                for tid, b in prior_bias.items():
                    if tid >= 0:
                        logits[:, tid] += b
 
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
            for i in range(E):
                vmt[i, :] = vmt[i, :] & ~finished[i]
 
            logits = logits.masked_fill(~vmt, -1e8)
            all_masked = (logits <= -1e7).all(dim=-1)
            logits[all_masked, 0] = 0.0
 
            log_p = F.log_softmax(logits, dim=-1)
            prob = torch.exp(log_p)
            ent = -(prob * log_p).sum(dim=-1)
            ent = torch.nan_to_num(ent, nan=0.0)
 
            act = actions_t[:, t + 1]
            nll = F.nll_loss(log_p, act, reduction='none', ignore_index=self.vocab.pad_id)
            nll = torch.nan_to_num(nll, nan=0.0)
 
            vp = torch.tensor([t < lengths[i] and not finished[i] for i in range(E)],
                              dtype=torch.bool, device=self.device)
            all_nll.append(nll)
            all_ent.append(ent)
            all_w.append(vp.float())
 
            for i in range(E):
                if finished[i]:
                    continue
                tid = act[i].item()
                if tid == self.vocab.pad_id:
                    finished[i] = True
                    continue
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
        # pg = pg / (pg.detach().abs().mean() + 1e-8)
        # ent_mean = ent_mean / (ent_mean.detach().abs().mean() + 1e-8)
        ent_loss = -self.entropy_weight * ent_mean
        return pg, ent_loss
 
    def _pqt_step(self, prior_bias=None) -> Optional[torch.Tensor]:
        n_pqt = max(1, int(self.pqt_mix_ratio * len(self.pqt_buffer)))
        samples = self.pqt_buffer.sample(n_pqt)
        if not samples:
            return None
        pseudo_programs = []
        for token_ids_tuple, prefix in samples:
            prog = Program.__new__(Program)
            prog.token_ids = list(token_ids_tuple)
            prog.prefix = prefix
            prog.vocab = self.vocab
            prog.config = self.policy.config
            prog._reward = None
            pseudo_programs.append(prog)
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