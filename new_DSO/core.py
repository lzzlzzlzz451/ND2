import os
import torch
import numpy as np
import logging
from typing import List
from .config import get_default_config
from .vocabulary import Vocabulary
from .valid_mask import ValidMaskComputer
from .policy import GNNTransformerPolicy
from .program import Program
from .reward import NDRewardSolver
from ND2.GDExpr import GDExprClass, GDExpr
from ND2.utils import AttrDict, seed_all
from .gp_controller import NDGPController
from .risk_seeking import RiskSeekingSelector
from .baseline import BaselineComputer
from .policy_optimizer import DSOOptimizer
 
logger = logging.getLogger('new_DSO')
 
 
class NewDSO:
    """
    new_DSO: 将 DSO 的风险寻求策略梯度方法适配到网络动力学场景。
 
    ★ 改造：策略网络使用 GNN Encoder + Transformer Decoder，
       图数据通过二值化管线编码为 graph_emb，作为解码器 memory。
    """
    def __init__(self, config=None, gdexpr_config=None,
                 Xv=None, Xe=None, A=None, G=None, Y=None, mask=None):
        """第零步：初始化所有核心组件。"""
        # 0.1 配置
        self.config = config or get_default_config()
        if gdexpr_config is not None:
            self.config = self._merge_gdexpr_config(self.config, gdexpr_config)
 
        # 0.2 随机种子
        seed_all(self.config.training.seed)
 
        # 0.3 词汇表
        self.vocab = Vocabulary(self.config)
        logger.info(f"[new_DSO] 词汇表大小: {self.vocab.n_words}")
 
        # 0.4 有效掩码计算器
        self.valid_mask_computer = ValidMaskComputer(self.vocab, self.config)
 
        # 0.5 策略网络 (GNN + Transformer)
        self.policy = GNNTransformerPolicy(self.vocab, self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        logger.info(f"[new_DSO] 策略网络参数量: "
                     f"{sum(p.numel() for p in self.policy.parameters()):,}")
        enc_params = sum(p.numel() for p in self.policy.GNN_encoder.parameters())
        dec_params = sum(p.numel() for p in self.policy.decoder.parameters())
        logger.info(f"[new_DSO]   编码器(GNN+Trans): {enc_params:,}  "
                     f"解码器(Trans): {dec_params:,}")
 
        # 0.6 GDExpr 表达式系统
        self.gdexpr = GDExprClass(self.config)
 
        # 0.7 数据存储
        self.Xv = Xv or {}
        self.Xe = Xe or {}
        self.A = A
        self.G = G
        self.Y = Y
        self.mask = mask
 
        # ★ 0.8 预计算图数据 embedding（一次性编码，采样时复用）
        self.graph_emb = None
        self.root_type = getattr(self.config.data, 'root_type', 'node')
        if Y is not None and A is not None and G is not None:
            self.graph_emb = self.encode_data(Xv, Xe, A, G, Y, self.root_type, mask)
            logger.info(f"[new_DSO] 图数据编码完成，graph_emb shape: {self.graph_emb.shape}")
 
        # 0.9 奖励计算器
        if Y is not None and A is not None and G is not None:
            self.reward_solver = NDRewardSolver(
                Xv=self.Xv, Xe=self.Xe, A=self.A, G=self.G,
                Y=self.Y, mask=self.mask,
                complexity_base=self.config.data.complexity_base,
                sample_num=self.config.data.sample_num,
                bfgs_max_iter=30,
                bfgs_full_iter=1000,
            )
        else:
            self.reward_solver = None
 
        # 0.10 训练状态
        self.n_evals = 0
        self.best_reward = -np.inf
        self.best_program = None
        self.best_metrics = {}
        self.start_time = None
 
        Program.clear_cache()
        Program.clear_reward_cache()
 
        # 0.11 GP 控制器
        gp_enabled = getattr(config, 'gp', None) is not None
        if gp_enabled and Y is not None:
            self.gp_controller = NDGPController(
                vocab=self.vocab,
                gdexpr=self.gdexpr,
                config=self.config,
                reward_solver=self.reward_solver,
                root_type=self.config.data.root_type,
            )
        else:
            self.gp_controller = None
 
        # 0.12 风险寻求选择器
        tc = config.training
        self.risk_selector = RiskSeekingSelector(
            epsilon=tc.epsilon,
            use_memory_queue=True,
            memory_queue_size=10,
            memory_decay=0.9,
            min_elite=max(1, tc.batch_size // 20),
        )
 
        logger.info(f"[new_DSO] GP-Meld: {'启用' if self.gp_controller else '禁用'}")
        logger.info(f"[new_DSO] GNN编码器: {'冻结' if self.config.encoder.freeze else '可训练'}")
 
        # 0.13 基线计算器
        tc = config.training
        self.baseline_computer = BaselineComputer(
            mode=tc.baseline_mode,
            ewma_decay=0.1,
            combined_alpha=0.7,
        )
 
        # ★ 0.14 优化器——只优化非冻结参数
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        tc = config.training
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
        self.optimizer.set_graph_emb(self.graph_emb)
        logger.info(f"[new_DSO] 可训练参数量: {sum(p.numel() for p in trainable_params):,}")
 
    # -------------------------------------------------------
    # ★ 图数据编码（二值化管线）
    # -------------------------------------------------------
    def encode_data(self, Xv, Xe, A, G, Y, root_type, mask=None):
        """
        使用 ND2 的二值化管线将图数据编码为 graph_emb。
 
        步骤：
        1. 将 Xv/Xe/Y 组织为 var_dict（与 ND2 的 NDformer.set_data 相同格式）
        2. 使用 GDExpr.parse_float 二值化
        3. 通过 GNN + Transformer Encoder 编码
        4. 缓存结果，采样时直接复用
 
        返回:
            graph_emb: (1, N_sample, d_model)，unsqueeze 了 batch 维度方便 expand
        """
        self.policy.eval()
        with torch.no_grad():
            V = A.shape[0]
            E = G.shape[0]
            T = Y.shape[0]
 
            # 构建 var_dict（与 ND2 的 set_data 逻辑一致）
            var_dict = dict(A=A, G=G, out=Y)
            for idx, (k, v) in enumerate(Xv.items(), 1):
                if v.ndim == 1:
                    v = v.reshape(1, -1).repeat(T, axis=0)
                if v.shape[-1] == 1:
                    v = v.repeat(V, axis=-1)
                var_dict[f'v{idx}'] = v
            for idx, (k, e) in enumerate(Xe.items(), 1):
                if e.ndim == 1:
                    e = e.reshape(1, -1).repeat(T, axis=0)
                if e.shape[-1] == 1:
                    e = e.repeat(E, axis=-1)
                var_dict[f'e{idx}'] = e
 
            # 构建 v/e 矩阵（与 ND2 的 NDformer.encode 相同逻辑）
            out = var_dict['out']
            N = out.shape[0]
            v = [out if root_type == 'node' else np.zeros((N, V))] + \
                [var_dict.get(var, np.zeros((N, V))) for var in GDExpr.variable.node]
            v = np.stack(v, axis=-1)  # (N, V, 1 + d_v)
 
            e = [out if root_type == 'edge' else np.zeros((N, E))] + \
                [var_dict.get(var, np.zeros((N, E))) for var in GDExpr.variable.edge]
            e = np.stack(e, axis=-1)  # (N, E, 1 + d_e)
 
            # 二值化
            v_bits = torch.from_numpy(GDExpr.parse_float(v)).to(
                self.device, torch.float32)  # (N, V, 1+d_v, 16)
            e_bits = torch.from_numpy(GDExpr.parse_float(e)).to(
                self.device, torch.float32)  # (N, E, 1+d_e, 16)
            G_t = torch.from_numpy(G).to(self.device, torch.long)
            A_t = torch.from_numpy(A).to(self.device, torch.long)
 
            # GNN + Transformer Encoder
            graph_emb = self.policy.encode_graph(
                v_bits, e_bits, G_t, A_t, root_type, mask=mask
            )  # (N_sample, d_model)
 
            # unsqueeze batch 维度：方便后续 expand 到 batch_size
            graph_emb = graph_emb.unsqueeze(0)  # (1, N_sample, d_model)
 
        return graph_emb
 
    # -------------------------------------------------------
    # 第七步：策略梯度更新
    # -------------------------------------------------------
    def policy_update(self, elite_programs, elite_rewards, baseline):
        prior_bias = self._get_prior_bias()
        return self.optimizer.update(
            elite_programs, elite_rewards, baseline, prior_bias
        )
 
    def select_elite(self, programs, rewards):
        return self.risk_selector.select(programs, rewards)
 
    def gp_evolve(self, programs: List[Program]) -> List[Program]:
        if self.gp_controller is None:
            return []
        gp_programs = self.gp_controller.evolve(programs)
        if self.reward_solver is not None:
            for p in gp_programs:
                _ = p.reward
        return gp_programs
 
    def _merge_gdexpr_config(self, dso_config, gdexpr_config):
        """将 GDExpr 所需的配置合并到 DSO 配置中"""
        merged = AttrDict(dict(dso_config))
        for key in ['max_complexity', 'max_coeff_num']:
            if key in gdexpr_config:
                merged[key] = gdexpr_config[key]
        return merged
 
    def _get_prior_bias(self):
        """先验偏置（可被子类覆盖）"""
        return None