import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vocabulary import Vocabulary
 
 
class TransformerPolicy(nn.Module):
    """
    基于 Transformer Decoder 的自回归策略网络，替代原来的 LSTM。
 
    每一步输入完整的 prefix 序列，通过因果自注意力输出下一步 logits。
    相比 RNN 的优势：直接 attend 所有历史 token，不依赖压缩的 hidden state。
    """
    def __init__(self, vocab: Vocabulary, config):
        super().__init__()
        self.vocab = vocab
        self.config = config
        pc = config.policy
 
        self._n_actions = max(vocab.word2id.values()) + 1
        self.max_length = pc.max_length
        self.d_model = pc.d_model
 
        # Token embedding
        self.token_embedding = nn.Embedding(self._n_actions, self.d_model,
                                            padding_idx=vocab.pad_id)
 
        # 位置编码（正弦余弦，不可学习）
        self.pos_encoding = self._build_sinusoidal_encoding(self.max_length + 1, self.d_model)
 
        # Dropout
        self.embed_dropout = nn.Dropout(pc.dropout)
 
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=pc.nhead,
            dim_feedforward=pc.dim_feedforward,
            dropout=pc.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN，训练更稳定
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=pc.num_layers,
        )
 
        # 用全零 memory 代替 encoder 输出（纯解码器模式）
        # memory 是 (1, B, d_model) 的可学习参数，模拟全局上下文
        self.memory_param = nn.Parameter(
            torch.randn(1, 1, self.d_model) * 0.02
        )
 
        # Layer Norm
        self.ln_f = nn.LayerNorm(self.d_model)
 
        # 输出头
        self.output_head = nn.Linear(self.d_model, self._n_actions)
 
        self._init_weights()
 
    def _init_weights(self):
        """Xavier 初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
 
    @staticmethod
    def _build_sinusoidal_encoding(max_len, d_model):
        """标准正弦余弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model)，不注册为 buffer 也没关系，推理时不会变
 
    def _generate_causal_mask(self, seq_len, device):
        """
        生成因果注意力掩码：位置 i 只能看到 ≤ i 的位置。
        返回 (seq_len, seq_len) 的 bool mask，True 表示禁止注意。
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
 
    def forward_full(self, prefix_ids):
        """
        给定完整 prefix 序列，输出每个位置的 logits。
 
        参数:
            prefix_ids: (B, L) int，SOS 开头的 token id 序列
 
        返回:
            logits: (B, L, n_actions) 每个位置的 logits
        """
        B, L = prefix_ids.shape
        device = prefix_ids.device
 
        # Token + Position embedding
        tok_emb = self.token_embedding(prefix_ids)  # (B, L, d_model)
        pos_emb = self.pos_encoding[:L].unsqueeze(0).to(device)  # (1, L, d_model)
        x = self.embed_dropout(tok_emb + pos_emb)
 
        # Causal mask
        causal_mask = self._generate_causal_mask(L, device)
 
        # Memory: (1, B, d_model) → expand to (1, B, d_model)
        memory = self.memory_param.expand(1, B, -1)  # (1, B, d_model)
 
        # Transformer decode
        x = self.decoder(
            tgt=x,                    # (B, L, d_model)
            memory=memory,            # (1, B, d_model)
            tgt_mask=causal_mask,     # (L, L)
            tgt_is_causal=True,
        )  # (B, L, d_model)
 
        x = self.ln_f(x)
        logits = self.output_head(x)  # (B, L, n_actions)
        return logits
 
    def forward_step(self, prefix_ids):
        """
        单步前向：给定 prefix，输出最后一个位置的 logits。
        兼容旧的 step-by-step 调用方式，但内部用完整序列前向。
 
        参数:
            prefix_ids: (B, L) int，SOS 开头的前缀序列
 
        返回:
            logits: (B, n_actions) 最后一个位置的 logits
        """
        all_logits = self.forward_full(prefix_ids)  # (B, L, n_actions)
        return all_logits[:, -1, :]  # (B, n_actions)
 
    def sample(self, batch_size, valid_mask_computer, device='cpu', prior_bias=None):
        """
        自回归采样一个 batch 的前缀表达式。
 
        返回:
            actions_batch: (B, L) int
            log_probs_batch: (B, L) float
            entropies_batch: (B, L) float
            finished_batch: (B,) bool
            masks_batch: (B, L) bool
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
 
            # 追踪每个序列的状态
            danglings = torch.ones(batch_size, dtype=torch.long, device=device)
            coeff_c = torch.zeros(batch_size, dtype=torch.long, device=device)
            coeff_cv = torch.zeros(batch_size, dtype=torch.long, device=device)
            coeff_ce = torch.zeros(batch_size, dtype=torch.long, device=device)
 
            # prefix 从 SOS 开始
            prefix = torch.full((batch_size, 1), self.vocab.sos_id,
                                dtype=torch.long, device=device)
 
            for t in range(max_L):
                if finished.all():
                    break
 
                # 前向传播：输入完整 prefix，取最后位置 logits
                logits = self.forward_step(prefix)  # (B, n_actions)
 
                # 加入先验偏置
                if prior_bias is not None:
                    for token_id, bias in prior_bias.items():
                        logits[:, token_id] += bias
 
                # 计算每个样本的有效掩码
                prefix_lists = []
                dangling_list = []
                coeff_list = []
                for i in range(batch_size):
                    if finished[i]:
                        prefix_lists.append([])
                        dangling_list.append(0)
                        coeff_list.append((0, 0, 0))
                    else:
                        # prefix 中去掉 SOS
                        valid_actions = prefix[i, 1:].cpu().tolist()
                        valid_actions = [a for a in valid_actions if a != self.vocab.pad_id]
                        prefix_lists.append(valid_actions)
                        dangling_list.append(danglings[i].item())
                        coeff_list.append((coeff_c[i].item(),
                                           coeff_cv[i].item(),
                                           coeff_ce[i].item()))
 
                has_var_list = has_var.cpu().tolist()
                valid_masks = valid_mask_computer.compute_mask_batch(
                    prefix_lists, dangling_list, coeff_list, has_variables=has_var_list
                )
 
                for i in range(batch_size):
                    if finished[i]:
                        valid_masks[i, :] = False
 
                valid_masks_t = torch.from_numpy(valid_masks).to(device)
                logits = logits.masked_fill(~valid_masks_t, -1e8)
 
                all_masked = (logits <= -1e7).all(dim=-1)
                logits[all_masked, 0] = 0.0
 
                # 计算概率分布
                probs = F.softmax(logits, dim=-1)
 
                # 采样
                dist = torch.distributions.Categorical(probs)
                sampled = dist.sample()  # (B,)
 
                step_log_probs = dist.log_prob(sampled)
                step_entropies = dist.entropy()
 
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
 
                # ★ 关键区别：拼接到 prefix，而非只传上一个 token
                prefix = torch.cat([prefix, sampled.unsqueeze(1)], dim=1)  # (B, t+2)
 
            return (actions_batch, log_probs_batch, entropies_batch,
                    finished, masks_batch)