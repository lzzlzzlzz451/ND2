import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vocabulary import Vocabulary
from ND2.model.utils import GNN
 
logger = logging.getLogger('new_DSO.Policy') 

class GNNTransformerPolicy(nn.Module):
    """
    GNN Encoder + Transformer Decoder 的策略网络。
 
    编码端：复用 ND2 的 GNN + Transformer Encoder，将图数据编码为 memory。
    解码端：Transformer Decoder，公式前缀 attend 图数据 memory，输出下一步 logits。
 
    与原 TransformerPolicy 的关键区别：
    - 原来用可学习 memory_param 模拟全局上下文（没有图信息）
    - 现在用 GNN 编码的真实 graph embedding 作为 memory（cross-attention）
    """
 
    def __init__(self, vocab: Vocabulary, config):
        super().__init__()
        self.vocab = vocab
        self.config = config
        pc = config.policy
        ec = config.encoder
 
        self._n_actions = max(vocab.word2id.values()) + 1
        self.max_length = pc.max_length
        self.d_model = pc.d_model
        self.gnn_d_model = 512
 
        # ============ GNN 编码器（复用 ND2） ============
        self.GNN_encoder = GNN(
            d_emb=self.gnn_d_model,
            n_layers=ec.n_GNN_layers,
            node_dim=ec.n_node_vars * ec.d_data_feat,
            edge_dim=ec.n_edge_vars * ec.d_data_feat,
            dropout=pc.dropout,
        )
        self.encoder_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=self.gnn_d_model,
            nhead=self.gnn_d_model // 64,
            dim_feedforward=self.gnn_d_model * 4,   # ← 这里！512*4=2048，不要用 pc.dim_feedforward
            dropout=pc.dropout,
            activation='gelu',
            batch_first=True,
        ), num_layers=ec.n_transformer_layers)

        self.graph_proj = nn.Sequential(
            nn.Linear(self.gnn_d_model, self.d_model),
            nn.GELU(),
        )
 
        self.encoder_ln = nn.LayerNorm(self.gnn_d_model)
 
        # 记录编码器配置，供 freeze 使用
        self._encoder_freeze = ec.freeze
 
        # ============ 解码器（原 Transformer Decoder） ============
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
 
        # Layer Norm
        self.ln_f = nn.LayerNorm(self.d_model)
 
        # 输出头
        self.output_head = nn.Linear(self.d_model, self._n_actions)
 
        self._init_weights()
 
        # ★ 冻结编码器
        if self._encoder_freeze:
            self.freeze_encoder()
 
    # -------------------------------------------------------
    # 编码器冻结 / 解冻
    # -------------------------------------------------------
    def freeze_encoder(self):
        """冻结 GNN 编码器参数，RL 训练时不更新"""
        for param in self.GNN_encoder.parameters():
            param.requires_grad = False
        for param in self.encoder_transformer.parameters():
            param.requires_grad = False
        for param in self.encoder_ln.parameters():
            param.requires_grad = False
 
    def unfreeze_encoder(self):
        """解冻 GNN 编码器参数（如果需要微调）"""
        for param in self.GNN_encoder.parameters():
            param.requires_grad = True
        for param in self.encoder_transformer.parameters():
            param.requires_grad = True
        for param in self.encoder_ln.parameters():
            param.requires_grad = True
 
    def load_from_ndformer(self, ckpt_path, device='cpu'):
        """
        从 NDformer 的 checkpoint.pth 加载 GNN + encoder Transformer 权重。
 
        checkpoint 结构（见 ND2/model/model.py:200-206）:
            {
                'encoder': encoder.state_dict(),  # keys: GNN.xxx, Transformer.xxx
                'decoder': ...,
                'optimizer': ...,
                'scheduler': ...,
            }
 
        映射关系:
            ckpt['encoder'] 的 'GNN.xxx'     → self.GNN_encoder 的 'xxx'
            ckpt['encoder'] 的 'Transformer.xxx' → self.encoder_transformer 的 'xxx'
        """
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except ModuleNotFoundError:
            # ND2 曾叫 src，兼容旧 checkpoint
            from io import BytesIO
            content = open(ckpt_path, 'rb').read()
            content = content.replace(b'src', b'ND2')
            ckpt = torch.load(BytesIO(content), map_location=device, weights_only=False)
 
        encoder_sd = ckpt['encoder']
 
        # ---- 拆分 GNN 和 Transformer 的 key ----
        gnn_keys = {k: v for k, v in encoder_sd.items() if k.startswith('GNN.')}
        transformer_keys = {k: v for k, v in encoder_sd.items() if k.startswith('Transformer.')}
        ln_keys = {k: v for k, v in encoder_sd.items() if k.startswith('norm.')}
 
        # ---- 加载 GNN：去掉 'GNN.' 前缀 ----
        if gnn_keys:
            gnn_sd = {k.replace('GNN.', '', 1): v for k, v in gnn_keys.items()}
            missing, unexpected = self.GNN_encoder.load_state_dict(gnn_sd, strict=False)
            if missing:
                logger.warning(f"[load_ndformer] GNN missing keys: {missing}")
            if unexpected:
                logger.warning(f"[load_ndformer] GNN unexpected keys: {unexpected}")
            logger.info(f"[load_ndformer] GNN 加载 {len(gnn_sd)} 个参数")
        else:
            logger.warning("[load_ndformer] checkpoint 中未找到 GNN 权重")
 
        # ---- 加载 encoder Transformer：去掉 'Transformer.' 前缀 ----
        if transformer_keys:
            tf_sd = {k.replace('Transformer.', '', 1): v for k, v in transformer_keys.items()}
            missing, unexpected = self.encoder_transformer.load_state_dict(tf_sd, strict=False)
            if missing:
                logger.warning(f"[load_ndformer] Transformer missing keys: {missing}")
            if unexpected:
                logger.warning(f"[load_ndformer] Transformer unexpected keys: {unexpected}")
            logger.info(f"[load_ndformer] encoder Transformer 加载 {len(tf_sd)} 个参数")
        else:
            logger.warning("[load_ndformer] checkpoint 中未找到 encoder Transformer 权重")
 
        # ---- LayerNorm（如果有的话）----
        if ln_keys:
            ln_sd = {k.replace('norm.', '', 1): v for k, v in ln_keys.items()}
            self.encoder_ln.load_state_dict(ln_sd, strict=False)
            logger.info(f"[load_ndformer] encoder LayerNorm 加载 {len(ln_sd)} 个参数")
 
        # ---- 冻结 ----
        if self._encoder_freeze:
            self.freeze_encoder()
 
        logger.info(f"[load_ndformer] 编码器权重加载完成，冻结={self._encoder_freeze}")
    # -------------------------------------------------------
    # 图数据编码
    # -------------------------------------------------------
    def encode_graph(self, v_bits, e_bits, G, A, root_type, mask=None):
        """
        GNN 编码图数据，输出 graph embedding。
 
        参数:
            v_bits: (N, V, max_node_vars_n, d_data_feat) 二值化节点特征
            e_bits: (N, E, max_edge_vars_n, d_data_feat) 二值化边特征
            G: (E, 2) 边列表
            A: (V, V) 邻接矩阵
            root_type: 'node' 或 'edge'
            mask: (N, V or E) 可选掩码
 
        返回:
            data_emb: (N_sample, d_model) 图数据的 embedding
        """
        ec = self.config.encoder
        # pad 到 max_node_vars_n / max_edge_vars_n
        v_bits = F.pad(v_bits, (0, 0, 0, ec.n_node_vars - v_bits.shape[2]), value=0.)
        e_bits = F.pad(e_bits, (0, 0, 0, ec.n_edge_vars - e_bits.shape[2]), value=0.)
        v_emb = v_bits.flatten(-2, -1)  # (N, V, n_node_vars * d_data_feat)
        e_emb = e_bits.flatten(-2, -1)  # (N, E, n_edge_vars * d_data_feat)
 
        # GNN message passing
        v_emb, e_emb = self.GNN_encoder(v_emb, e_emb, G, A)
 
        # 选择 node 或 edge embedding 作为输出
        data_emb = v_emb if root_type == 'node' else e_emb  # (N, V or E, d_model)
 
        if mask is not None:
            data_emb = data_emb[mask, :]
        else:
            data_emb = data_emb.flatten(0, 1)  # (N * V/E, d_model)
 
        # 采样控制
        ec = self.config.encoder
        if data_emb.shape[0] > ec.max_sample_num:
            idx = torch.randperm(data_emb.shape[0])[:ec.max_sample_num]
            data_emb = data_emb[idx]
 
        # Transformer Encoder 做全局聚合
        data_emb = self.encoder_transformer(data_emb)  # (N_sample, d_model)
        data_emb = self.encoder_ln(data_emb)
        data_emb = self.graph_proj(data_emb)    # (N_sample, 128) ← 投影
 
        return data_emb
 
    # -------------------------------------------------------
    # 初始化
    # -------------------------------------------------------
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
        return pe
 
    def _generate_causal_mask(self, seq_len, device):
        """生成因果注意力掩码"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
 
    # -------------------------------------------------------
    # 前向传播
    # -------------------------------------------------------
    def forward_full(self, prefix_ids, graph_emb=None):
        """
        给定完整 prefix 序列 + 图 embedding，输出每个位置的 logits。
 
        参数:
            prefix_ids: (B, L) int，SOS 开头的 token id 序列
            graph_emb: (1, N, d_model) 预计算的图 embedding，expand 到 batch
 
        返回:
            logits: (B, L, n_actions)
        """
        B, L = prefix_ids.shape
        device = prefix_ids.device
 
        # Token + Position embedding
        tok_emb = self.token_embedding(prefix_ids)  # (B, L, d_model)
        pos_emb = self.pos_encoding[:L].unsqueeze(0).to(device)
        x = self.embed_dropout(tok_emb + pos_emb)
 
        # Causal mask
        causal_mask = self._generate_causal_mask(L, device)
 
        # ★ 关键改动：用真实 graph_emb 替代 memory_param
        if graph_emb is not None:
            memory = graph_emb.expand(B, -1, -1)  # (B, N, d_model)
        else:
            # 回退：如果没有图数据（调试用），用零向量
            memory = torch.zeros(1, B, self.d_model, device=device)
 
        # Transformer decode：公式 attend 图数据
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
        )
 
        x = self.ln_f(x)
        logits = self.output_head(x)
        return logits
 
    def forward_step(self, prefix_ids, graph_emb=None):
        """
        单步前向：给定 prefix，输出最后一个位置的 logits。
        """
        all_logits = self.forward_full(prefix_ids, graph_emb)
        return all_logits[:, -1, :]
 
    def sample(self, batch_size, valid_mask_computer, device='cpu',
               prior_bias=None, graph_emb=None):
        """
        自回归采样一个 batch 的前缀表达式。
 
        参数:
            batch_size: int
            valid_mask_computer: ValidMaskComputer
            device: str
            prior_bias: dict or None
            graph_emb: (1, N, d_model) 预计算的图 embedding
 
        返回:
            (actions_batch, log_probs_batch, entropies_batch,
             finished, masks_batch, probs_batch)
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
            probs_batch = torch.zeros(batch_size, max_L, self._n_actions, device=device)
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
 
                # ★ 传入 graph_emb
                logits = self.forward_step(prefix, graph_emb=graph_emb)
 
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
                sampled = dist.sample()
 
                step_log_probs = dist.log_prob(sampled)
                step_entropies = dist.entropy()
 
                for i in range(batch_size):
                    if not finished[i]:
                        actions_batch[i, t] = sampled[i]
                        log_probs_batch[i, t] = step_log_probs[i]
                        entropies_batch[i, t] = step_entropies[i]
                        masks_batch[i, t] = True
                        probs_batch[i, t] = probs[i]
                        tid = sampled[i].item()
                        danglings[i] = danglings[i] - 1 + self.vocab.arity(tid)
                        if self.vocab.kind(tid) == 'coefficient': coeff_c[i] += 1
                        elif self.vocab.kind(tid) == 'node_coeff': coeff_cv[i] += 1
                        elif self.vocab.kind(tid) == 'edge_coeff': coeff_ce[i] += 1
                        if self.vocab.kind(tid) == 'variable': has_var[i] = True
                        if danglings[i] <= 0: finished[i] = True
 
                prefix = torch.cat([prefix, sampled.unsqueeze(1)], dim=1)
 
            return (actions_batch, log_probs_batch, entropies_batch,
                    finished, masks_batch, probs_batch)
 
 
# ============ 向后兼容别名 ============
# 原来用 RNVPolicy / TransformerPolicy 的代码无需修改 import
RNVPolicy = GNNTransformerPolicy
TransformerPolicy = GNNTransformerPolicy