import numpy as np
import logging
from collections import deque
from typing import List, Tuple, Optional
from .program import Program
 
logger = logging.getLogger('new_DSO.RiskSeeking')
 
 
class MemoryQueue:
    """
    历史奖励队列，用于稳定分位数估计。
 
    DSO 原版中，如果启用 Memory Queue，分位数估计会结合历史样本的
    加权分布，避免因单 batch 采样噪声导致的分位数跳变。
 
    存储 (reward, weight) 对，越近期的 batch 权重越高。
    """
    def __init__(self, max_size: int = 10, decay: float = 0.9):
        self.max_size = max_size
        self.decay = decay            # 历史衰减系数
        self._queue = deque(maxlen=max_size)
        self._weights = deque(maxlen=max_size)
 
    def push(self, rewards: np.ndarray):
        """推入一个 batch 的奖励"""
        self._queue.append(rewards.copy())
        self._weights.append(1.0)  # 最新 batch 权重为 1
        # 衰减历史权重
        for i in range(len(self._weights) - 1):
            self._weights[i] *= self.decay
 
    def get_weighted_rewards(self) -> Optional[np.ndarray]:
        """获取加权后的历史奖励样本"""
        if not self._queue:
            return None
        all_rewards = []
        all_weights = []
        for rewards, w in zip(self._queue, self._weights):
            # 按 w 比例采样（近似加权）
            n_samples = max(1, int(len(rewards) * w / sum(self._weights) * 100))
            indices = np.random.choice(len(rewards), min(n_samples, len(rewards)),
                                       replace=True)
            all_rewards.append(rewards[indices])
            all_weights.append(np.full(len(indices), w))
        return np.concatenate(all_rewards)
 
    def clear(self):
        self._queue.clear()
        self._weights.clear()
 
    def __len__(self):
        return len(self._queue)
 
 
class RiskSeekingSelector:
    """
    第五步：风险寻求策略梯度——筛选精英样本。
 
    核心逻辑:
    1. 计算分位数: quantile = np.quantile(r, 1 - epsilon)
    2. 筛选: 只保留 r >= quantile 的样本（即 top-ε 的精英）
    3. 若启用 Memory Queue，分位数估计结合历史样本，更稳定
 
    参数:
        epsilon: float, 精英比例（默认 0.05 = top-5%）
        use_memory_queue: bool, 是否使用历史队列
        memory_queue_size: int, 历史队列最大长度
        memory_decay: float, 历史衰减系数
        min_elite: int, 最少精英数量（避免 batch 太小时无人入选）
    """
    def __init__(self,
                 epsilon: float = 0.05,
                 use_memory_queue: bool = True,
                 memory_queue_size: int = 10,
                 memory_decay: float = 0.9,
                 min_elite: int = 1):
        self.epsilon = epsilon
        self.use_memory_queue = use_memory_queue
        self.min_elite = min_elite
        self.memory_queue = MemoryQueue(
            max_size=memory_queue_size,
            decay=memory_decay
        )
 
    def select(self,
               programs: List[Program],
               rewards: np.ndarray) -> Tuple[List[Program], np.ndarray, float]:
        """
        第五步入口：从当前 batch 中筛选精英样本。
 
        参数:
            programs: 当前 batch 的 Program 列表（RL + GP 合并后）
            rewards: 对应的奖励数组, shape=(B,)
 
        返回:
            elite_programs: 精英 Program 列表
            elite_rewards: 精英奖励数组
            baseline: 基线值（= quantile，供第六步使用）
 
        流程:
        ┌─────────────────────────────────────────┐
        │ 1. 将当前 batch rewards 推入 MemoryQueue  │
        │ 2. 获取加权历史样本（若启用）               │
        │ 3. 计算 quantile = Q(1 - ε)              │
        │ 4. 筛选 r >= quantile 的样本              │
        │ 5. 保证最少 min_elite 个精英              │
        └─────────────────────────────────────────┘
        """
        assert len(programs) == len(rewards), \
            f"programs ({len(programs)}) != rewards ({len(rewards)})"
 
        # 1. 推入历史队列
        if self.use_memory_queue:
            self.memory_queue.push(rewards)
 
        # 2. 计算分位数
        quantile = self._compute_quantile(rewards)
 
        # 3. 筛选精英
        elite_mask = rewards >= quantile
 
        # 4. 保证最少精英数量
        n_elite = elite_mask.sum()
        if n_elite < self.min_elite:
            # 按奖励降序，取前 min_elite 个
            top_k = min(self.min_elite, len(rewards))
            elite_indices = np.argsort(rewards)[-top_k:]
            elite_mask = np.zeros(len(rewards), dtype=bool)
            elite_mask[elite_indices] = True
            n_elite = elite_mask.sum()
 
        elite_programs = [p for p, m in zip(programs, elite_mask) if m]
        elite_rewards = rewards[elite_mask]
 
        # baseline = quantile（第六步直接使用）
        baseline = quantile
 
        logger.info(
            f"[RiskSeeking] ε={self.epsilon:.2f} | "
            f"quantile={quantile:.4f} | "
            f"elite: {n_elite}/{len(rewards)} | "
            f"elite_reward: [{elite_rewards.min():.4f}, {elite_rewards.max():.4f}]"
        )
 
        return elite_programs, elite_rewards, baseline
 
    def _compute_quantile(self, current_rewards: np.ndarray) -> float:
        """
        计算风险寻求分位数。
 
        两种模式:
        - 不使用 MemoryQueue: quantile = Q(current_rewards, 1-ε)
        - 使用 MemoryQueue: 结合历史样本的加权分布
          quantile = Q(weighted_historical + current_rewards, 1-ε)
        """
        if self.use_memory_queue:
            historical = self.memory_queue.get_weighted_rewards()
            if historical is not None and len(historical) > 0:
                combined = np.concatenate([current_rewards, historical])
            else:
                combined = current_rewards
        else:
            combined = current_rewards
 
        # 过滤非有限值
        valid = combined[np.isfinite(combined)]
        if len(valid) == 0:
            return 0.0
 
        quantile = np.quantile(valid, 1 - self.epsilon)
        return quantile