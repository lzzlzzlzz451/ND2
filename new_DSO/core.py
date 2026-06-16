import os
import torch
import numpy as np
import logging
from .config import get_default_config
from .vocabulary import Vocabulary
from .valid_mask import ValidMaskComputer
from .policy import RNVPolicy
from .program import Program
from .reward import NDRewardSolver
from ND2.GDExpr import GDExprClass
from ND2.utils import AttrDict, seed_all
from .gp_controller import NDGPController
from .risk_seeking import RiskSeekingSelector
from .baseline import BaselineComputer
from .policy_optimizer import DSOOptimizer
 
logger = logging.getLogger('new_DSO')
 
 
class NewDSO:
    """
    new_DSO: 将 DSO 的风险寻求策略梯度方法适配到网络动力学场景。
 
    第零步: 初始化
    第一步: RNN 自回归采样表达式
    第二步: 补全与构建 Program 对象
 
    后续步骤（第三步~第八步）将在后续实现。
    """
    def __init__(self, config=None, gdexpr_config=None,
                 Xv=None, Xe=None, A=None, G=None, Y=None, mask=None):
        """
        第零步：初始化所有核心组件。
        """
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
 
        # 0.5 策略网络 (RNN)
        self.policy = RNVPolicy(self.vocab, self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        logger.info(f"[new_DSO] 策略网络参数量: "
                     f"{sum(p.numel() for p in self.policy.parameters()):,}")
 
        # 0.6 GDExpr 表达式系统（复用 ND2 的表达式求值和参数拟合）
        self.gdexpr = GDExprClass(self.config)
 
        # 0.7 数据存储（用于后续奖励计算）
        self.Xv = Xv or {}
        self.Xe = Xe or {}
        self.A = A
        self.G = G
        self.Y = Y
        self.mask = mask
 
         # ★ 0.9 奖励计算器（第三步新增）
        if Y is not None and A is not None and G is not None:
            self.reward_solver = NDRewardSolver(
                Xv=self.Xv, Xe=self.Xe, A=self.A, G=self.G,
                Y=self.Y, mask=self.mask,
                complexity_base=self.config.data.complexity_base,
                sample_num=self.config.data.sample_num,
                bfgs_max_iter=30,        # 采样阶段快速优化
                bfgs_full_iter=1000,     # 精确阶段充分优化
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
        logger.info("[new_DSO] 第零步初始化完成。")

        # ★ 0.11 GP 控制器（第四步新增）
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
        
        # ★ 0.12 风险寻求选择器（第五步新增）
        tc = config.training
        self.risk_selector = RiskSeekingSelector(
            epsilon=tc.epsilon,
            use_memory_queue=True,
            memory_queue_size=10,
            memory_decay=0.9,
            min_elite=max(1, tc.batch_size // 20),  # 至少保留 batch 的 5%
        )
 
        logger.info(f"[new_DSO] GP-Meld: {'启用' if self.gp_controller else '禁用'}")

        # ★ 0.13 基线计算器（第六步新增）
        tc = config.training
        self.baseline_computer = BaselineComputer(
            mode=tc.baseline_mode,        # 默认 'R_e'
            ewma_decay=0.1,
            combined_alpha=0.7,
        )

        tc = config.training
        self.optimizer = DSOOptimizer(
            policy=self.policy,
            vocab=self.vocab,
            valid_mask_computer=self.valid_mask_computer,
            device=self.device,
            entropy_weight=tc.entropy_weight,
            clip_grad_norm=100.0,
            learning_rate=tc.learning_rate,
            mode='PG',              # 'PG' 或 'PQT'
            pqt_max_size=50,
            pqt_mix_ratio=0.2,
        )
    
    # -------------------------------------------------------
    # ★ 第七步：策略梯度更新
    # -------------------------------------------------------
    def policy_update(self, elite_programs, elite_rewards, baseline):
        """
        第七步：对 elite 样本执行策略梯度更新。
 
        loss = pg_loss + entropy_loss
        pg_loss    = mean((r - baseline) * neglogp)
        entropy_loss = -α * mean(entropy)
 
        返回:
            stats: dict
        """
        prior_bias = self._get_prior_bias()
        return self.optimizer.update(
            elite_programs, elite_rewards, baseline, prior_bias
        )
    
    def select_elite(self, programs, rewards):
        """
        第五步：从 RL+GP 合并种群中筛选精英样本。
 
        只有 top-ε 的表达式参与后续策略更新。
 
        返回:
            elite_programs: 精英 Program 列表
            elite_rewards: 精英奖励数组
            baseline: 基线值（= quantile，第六步使用）
        """
        return self.risk_selector.select(programs, rewards)
    
    def gp_evolve(self, programs: List[Program]) -> List[Program]:
        """
        第四步：对 RNN 采样的种群执行 GP 进化。
 
        流程:
        1. 将 RNN 采样的 batch 作为种子种群
        2. 锦标赛选择 → 类型匹配交叉 + decompose 变异
        3. GP 新个体与 RL 采样在后续步骤统一进入策略梯度更新
 
        返回:
            gp_programs: GP 产生的新 Program 列表
        """
        if self.gp_controller is None:
            return []
 
        gp_programs = self.gp_controller.evolve(programs)
 
        # 对 GP 产生的个体计算奖励
        if self.reward_solver is not None:
            for p in gp_programs:
                _ = p.reward  # 触发惰性计算
 
        return gp_programs
 
    def _merge_gdexpr_config(self, dso_config, gdexpr_config):
        """将 GDExpr 所需的配置合并到 DSO 配置中"""
        merged = AttrDict(dict(dso_config))
        for key in ['max_complexity', 'max_coeff_num']:
            if key in gdexpr_config:
                merged.policy[key] = gdexpr_config[key]
        return merged
 
 
    def _get_prior_bias(self):
        """
        计算先验偏置（类似 DSO 的 Prior）。
        对 ND 图算子施加类型约束偏置：
        - 在 node 上下文中，提升 node 变量 / 抑制 edge 变量
        - 在 edge 上下文中，提升 edge 变量 / 抑制 node 变量
        - 对复杂度过高的表达式施加惩罚
        """
        bias = {}
        # 简单先验：稍微抑制 <C> 以避免系数过多
        bias[self.vocab.word2id.get('<C>', -1)] = -0.5
        return bias
 
    def step(self):
        """完整 DSO 迭代（第零步到第六步）"""
        # 第一步+第二步
        programs, actions, log_probs, entropies, masks = self.sample_batch()
 
        # 第三步
        rewards = self.compute_rewards(programs)
 
        # 第四步
        gp_programs = self.gp_evolve(programs)
        gp_rewards = np.array([p.reward for p in gp_programs]) if gp_programs else np.array([])
 
        all_programs = programs + gp_programs
        all_rewards = np.concatenate([rewards, gp_rewards]) if len(gp_rewards) > 0 else rewards
 
        # 第五步
        elite_programs, elite_rewards, quantile = self.select_elite(all_programs, all_rewards)
 
        # ★ 第六步：计算基线
        baseline = self.compute_baseline(elite_rewards, all_rewards, quantile)
 
        logger.info(
            f"[new_DSO] elite={len(elite_programs)} | "
            f"quantile={quantile:.4f} | baseline={baseline:.4f} | "
            f"best={self.best_reward:.4f}"
        )
 
        return (all_programs, all_rewards,
                elite_programs, elite_rewards,
                quantile, baseline,
                actions, log_probs, entropies, masks)

    def compute_rewards(self, programs):
        """
        第三步：对每个 Program 触发奖励计算。
 
        DSO 的核心设计是惰性计算：调用 p.reward 时才触发 BFGS 优化。
        这里我们显式遍历触发，同时：
        1. 对非终端表达式返回 reward=0
        2. 对含系数的表达式执行 BFGS 优化
        3. 若并行池可用，通过 pool.map 并行执行
 
        返回:
            rewards: np.ndarray, shape=(B,)
        """
        rewards = np.array([p.reward for p in programs])
 
        # 发现新的全局最优 → 精确重算
        if rewards.max() > self.best_reward:
            for i, p in enumerate(programs):
                if rewards[i] > self.best_reward:
                    # 精确评估（全量数据 + 更多 BFGS 迭代）
                    precise_reward = p.recompute_reward_precise()
                    rewards[i] = precise_reward
 
                    if precise_reward > self.best_reward:
                        self.best_reward = precise_reward
                        self.best_program = p
                        self.best_metrics = self.reward_solver.evaluate(
                            p.prefix_with_coef, {}
                        ) if self.reward_solver else {}
                        logger.note(
                            f"[new_DSO] ★ 新最优! reward={self.best_reward:.6f} | "
                            f"R²={self.best_metrics.get('R2', 'N/A')} | "
                            f"expr={p.prefix_with_coef}"
                        )
 
        return rewards

def step(self):
        """完整 DSO 迭代（第零步到第七步）"""
        # 第一步+第二步
        programs, actions, log_probs, entropies, masks = self.sample_batch()
 
        # 第三步
        rewards = self.compute_rewards(programs)
 
        # 第四步
        gp_programs = self.gp_evolve(programs)
        gp_rewards = np.array([p.reward for p in gp_programs]) if gp_programs else np.array([])
 
        all_programs = programs + gp_programs
        all_rewards = np.concatenate([rewards, gp_rewards]) if len(gp_rewards) > 0 else rewards
 
        # 第五步
        elite_programs, elite_rewards, quantile = self.select_elite(all_programs, all_rewards)
 
        # 第六步
        baseline = self.compute_baseline(elite_rewards, all_rewards, quantile)
 
        # ★ 第七步：策略梯度更新
        stats = self.policy_update(elite_programs, elite_rewards, baseline)
 
        return (all_programs, all_rewards,
                elite_programs, elite_rewards,
                quantile, baseline, stats)

# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    import numpy as np
 
    # 1. 准备配置
    config = get_default_config()
    config.data.root_type = 'node'
    config.policy.max_length = 20
 
    # 2. 准备数据（以 SIS 模型为例）
    # 假设已有图结构 G 和动态数据
    V, E = 50, 200
    G = np.random.randint(0, V, (E, 2))
    A = np.zeros((V, V))
    for e in G:
        A[e[0], e[1]] = 1
 
    # 节点变量: v1 (当前状态), 边变量: e1 (边权重)
    Xv = {'v1': np.random.randn(100, V)}
    Xe = {'e1': np.random.randn(100, E)}
    Y = np.random.randn(100, V)  # 目标: dv1/dt
 
    # 3. 初始化 new_DSO（第零步）
    dso = NewDSO(config=config, Xv=Xv, Xe=Xe, A=A, G=G, Y=Y)
 
    # 4. 采样一个 batch（第一步 + 第二步）
    programs, actions, log_probs, entropies, masks = dso.step()
 
    # # 5. 查看采样结果
    for i, p in enumerate(programs[:5]):
        print(f"  Program {i}: {p.prefix}  (terminal={p.is_terminal()})")