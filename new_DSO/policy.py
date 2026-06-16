import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vocabulary import Vocabulary   # ← 加这一行
 
class RNVPolicy(nn.Module):
    """
    基于 LSTM/GRU 的自回归策略网络，用于逐步采样前缀表达式的 token。
 
    与 ND2 的 Transformer 解码器不同，这里采用 DSO 风格的 RNN：
    - 每一步输入当前观察（上一步的 action embedding + 位置编码 + 先验偏置）
    - 输出所有 token 的 logits
    - 结合 valid mask 进行合法采样
 
    关键创新：融入 ND 图算子的类型约束作为先验偏置
    """
    def __init__(self, vocab: Vocabulary, config):
        super().__init__()
        self.vocab = vocab
        self.config = config
        pc = config.policy
 
        self.n_words = vocab.n_words
        self._n_actions = max(vocab.word2id.values()) + 1
        self.hidden_size = pc.hidden_size
        self.num_layers = pc.num_layers
        self.max_length = pc.max_length
        self.embedding_size = pc.embedding_size
 
        # Token embedding
        self.token_embedding = nn.Embedding(self._n_actions, self.embedding_size,
                                            padding_idx=vocab.pad_id)
 
        # 位置编码（可学习）
        self.position_embedding = nn.Embedding(self.max_length + 1, self.embedding_size)
 
        # RNN cell
        if pc.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=False,
            )
        elif pc.cell_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=False,
            )
        else:
            raise ValueError(f"Unsupported cell type: {pc.cell_type}")
 
        # 输出头: logits
        self.output_head = nn.Linear(self.hidden_size, self._n_actions)
 
        # 初始观察 embedding (t=0 时使用)
        self.initial_obs_embedding = nn.Parameter(
            torch.randn(1, 1, self.embedding_size) * 0.01
        )
 
    def forward(self, actions, hidden=None):
        """
        单步前向传播。
 
        参数:
            actions: (B,) 上一步采样的 token id，t=0 时为 SOS
            hidden: RNN 隐状态，t=0 时为 None
 
        返回:
            logits: (B, n_words)
            new_hidden: 更新后的隐状态
        """
        # (B,) -> (1, B, embedding_size)
        emb = self.token_embedding(actions).unsqueeze(0)
        output, new_hidden = self.rnn(emb, hidden)
        # output: (1, B, hidden_size)
        logits = self.output_head(output.squeeze(0))  # (B, n_words)
        return logits, new_hidden
 
    def sample(self, batch_size, valid_mask_computer, device='cpu', prior_bias=None):
        """
        自回归采样一个 batch 的前缀表达式。
 
        参数:
            batch_size: int
            valid_mask_computer: ValidMaskComputer
            device: str
            prior_bias: 可选的先验偏置 dict(token_id -> bias)
 
        返回:
            actions_batch: (B, L) int, 采样的 token id 序列
            log_probs_batch: (B, L) float, 每个 token 的对数概率
            entropies_batch: (B, L) float, 每个 token 的熵
            finished_batch: (B,) bool, 表达式是否完整
            masks_batch: (B, L) bool, 有效位置掩码（含 padding）
        """
        self.eval()
        with torch.no_grad():
            max_L = self.max_length
 
            # 初始化存储
            actions_batch = torch.full((batch_size, max_L), self.vocab.pad_id,
                                       dtype=torch.long, device=device)
            log_probs_batch = torch.zeros(batch_size, max_L, device=device)
            entropies_batch = torch.zeros(batch_size, max_L, device=device)
            masks_batch = torch.zeros(batch_size, max_L, dtype=torch.bool, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            has_var = torch.zeros(batch_size, dtype=torch.bool, device=device)
 
            # 初始输入: SOS token
            current_actions = torch.full((batch_size,), self.vocab.sos_id,
                                          dtype=torch.long, device=device)
            hidden = None
 
            # 追踪每个序列的悬空节点数和系数计数
            danglings = torch.ones(batch_size, dtype=torch.long, device=device)
            coeff_c = torch.zeros(batch_size, dtype=torch.long, device=device)
            coeff_cv = torch.zeros(batch_size, dtype=torch.long, device=device)
            coeff_ce = torch.zeros(batch_size, dtype=torch.long, device=device)
 
            # 逐步采样
            for t in range(max_L):
                if finished.all():
                    break
 
                # 前向传播得到 logits
                logits, hidden = self.forward(current_actions, hidden)  # (B, n_words)
 
                # 加入先验偏置
                if prior_bias is not None:
                    for token_id, bias in prior_bias.items():
                        logits[:, token_id] += bias
 
                # 计算每个样本的有效掩码
                # （将 tensor 转为 numpy 来调用 valid_mask_computer）
                prefix_lists = []
                dangling_list = []
                coeff_list = []
                for i in range(batch_size):
                    if finished[i]:
                        prefix_lists.append([])
                        dangling_list.append(0)
                        coeff_list.append((0, 0, 0))
                    else:
                        # 取当前已采样的有效 token
                        valid_actions = actions_batch[i, :t].cpu().tolist()
                        valid_actions = [a for a in valid_actions
                                         if a != self.vocab.pad_id]
                        prefix_lists.append(valid_actions)
                        dangling_list.append(danglings[i].item())
                        coeff_list.append((coeff_c[i].item(),
                                           coeff_cv[i].item(),
                                           coeff_ce[i].item()))
                
                has_var_list = has_var.cpu().tolist()
                valid_masks = valid_mask_computer.compute_mask_batch(
                    prefix_lists, dangling_list, coeff_list, has_variables=has_var_list
                )  # (B, n_words), numpy bool
 
                # 已完成的样本掩码全 False
                for i in range(batch_size):
                    if finished[i]:
                        valid_masks[i, :] = False
 
                valid_masks_t = torch.from_numpy(valid_masks).to(device)
 
                # 将不合法 token 的 logits 设为 -inf
                logits = logits.masked_fill(~valid_masks_t, -1e8)

                all_masked = (logits <= -1e7).all(dim=-1)
                logits[all_masked, 0] = 0.0  # 给个 dummy，反正不会用到
 
                # 计算概率分布
                probs = F.softmax(logits, dim=-1)  # (B, n_words)

                with open(f'log/step_{t}_probs.log', 'w') as f:
                    for i in range(batch_size):
                        if finished[i]:
                            continue
                        current_type = valid_mask_computer._get_current_type(prefix_lists[i])
                        f.write(f'[Step {t} Sample {i}] expected_type={current_type}\n')
                        f.write(f'[Step {t} Sample {i}] all probs:\n')
                        for tid in range(self.vocab.n_words):
                            p = probs[i, tid].item()
                            if p > 0:
                                f.write(f'  {self.vocab.id2word[tid]}: {p:.4f}\n')
                        # ★ 加这段：打印 mask 中允许的 token
                        vm = valid_masks[i]
                        allowed = [self.vocab.id2word[tid] for tid in range(self.vocab.n_words) if vm[tid]]
                        f.write(f'[Step {t} Sample {i}] mask_allowed: {allowed}\n')
 
                # 采样
                dist = torch.distributions.Categorical(probs)
                sampled = dist.sample()  # (B,)
 
                # 记录
                step_log_probs = dist.log_prob(sampled)   # (B,)
                step_entropies = dist.entropy()           # (B,)
                
                for i in range(batch_size):
                    if not finished[i]:
                        actions_batch[i, t] = sampled[i]
                        log_probs_batch[i, t] = step_log_probs[i]
                        entropies_batch[i, t] = step_entropies[i]
                        masks_batch[i, t] = True
                        tid = sampled[i].item()
                        danglings[i] = danglings[i] - 1 + self.vocab.arity(tid)
                        if self.vocab.kind(tid) == 'coefficient': coeff_c[i] += 1
                        elif self.vocab.kind(tid) == 'node_coeff': coeff_cv[i] += 1
                        elif self.vocab.kind(tid) == 'edge_coeff': coeff_ce[i] += 1
                        if self.vocab.kind(tid) == 'variable': has_var[i] = True
                        if danglings[i] <= 0: finished[i] = True
 
                current_actions = sampled
 
            return (actions_batch, log_probs_batch, entropies_batch,
                    finished, masks_batch)
