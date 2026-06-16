# ============================================================
# new_DSO/baseline.py — 策略梯度基线计算器
# ============================================================
import numpy as np
import logging
from typing import List, Optional
from .program import Program
 
logger = logging.getLogger('new_DSO.Baseline')
 
 
class BaselineComputer:
    """
    第六步：计算策略梯度的基线，降低方差。
 
    DSO 提供四种基线模式:
    ┌──────────┬───────────────────────────────────────────────────┐
    │ 模式     │ 公式                                              │
    ├──────────┼───────────────────────────────────────────────────┤
    │ R_e      │ b = quantile（默认，最激进）                        │
    │ ewma_R   │ b = EWMA(历史所有 reward 的均值)                   │
    │ ewma_R_e │ b = EWMA(历史所有 quantile)                       │
    │ combined │ b = α * quantile + (1-α) * ewma_R                │
    └──────────┴───────────────────────────────────────────────────┘
 
    默认模式 R_e 的直觉:
    - baseline = quantile 意味着只有超过分位数的 elite 才有正优势
    - 优势 = r - baseline，elite 的优势 > 0 → 增大其 action 概率
    - 这与第五步的筛选逻辑完全对齐：被丢弃的样本根本不参与计算
 
    其他模式更保守:
    - ewma_R: 用全局均值做基线，更多样本有正优势（类似普通 PG）
    - ewma_R_e: 用历史分位数做基线，比 R_e 更稳定
    - combined: 折中方案
 
    参数:
        mode: str, 'R_e' | 'ewma_R' | 'ewma_R_e' | 'combined'
        ewma_decay: float, EWMA 衰减系数（越小越平滑）
        combined_alpha: float, combined 模式中 quantile 的权重
    """
    def __init__(self,
                 mode: str = 'R_e',
                 ewma_decay: float = 0.1,
                 combined_alpha: float = 0.7):
        self.mode = mode
        self.ewma_decay = ewma_decay
        self.combined_alpha = combined_alpha
 
        # EWMA 状态
        self._ewma_R = None       # 历史均值的 EWMA
        self._ewma_R_e = None     # 历史分位数的 EWMA
        self._step_count = 0
 
    def compute(self, elite_rewards, all_rewards=None, quantile=None):
        # 先更新 EWMA
        if all_rewards is not None:
            m = np.nanmean(all_rewards[np.isfinite(all_rewards)]) if np.any(np.isfinite(all_rewards)) else 0.0
            self._ewma_R = m if self._ewma_R is None else (1-self.ewma_decay)*self._ewma_R + self.ewma_decay*m
        
        if quantile is not None and np.isfinite(quantile):
            self._ewma_R_e = quantile if self._ewma_R_e is None else (1-self.ewma_decay)*self._ewma_R_e + self.ewma_decay*quantile
        
        if self.mode == 'R_e':
            return quantile if quantile is not None else 0.0
        elif self.mode == 'ewma_R':
            return self._ewma_R if self._ewma_R is not None else 0.0
        elif self.mode == 'ewma_R_e':
            return self._ewma_R_e if self._ewma_R_e is not None else 0.0
        elif self.mode == 'combined':
            q = quantile if quantile is not None else 0.0
            e = self._ewma_R if self._ewma_R is not None else 0.0
            return self.combined_alpha * q + (1-self.combined_alpha) * e
 
    def _compute_by_mode(self, quantile: Optional[float]) -> float:
        """按模式计算基线"""
        if self.mode == 'R_e':
            # 默认模式：直接用分位数
            # 这是最激进的：只有 r > quantile 的 elite 才有正优势
            return quantile if quantile is not None else 0.0
 
        elif self.mode == 'ewma_R':
            # 用历史均值的 EWMA
            # 更保守：更多样本可能有正优势
            return self._ewma_R if self._ewma_R is not None else 0.0
 
        elif self.mode == 'ewma_R_e':
            # 用历史分位数的 EWMA
            # 比 R_e 更稳定，避免单 batch 分位数跳变
            return self._ewma_R_e if self._ewma_R_e is not None else 0.0
 
        elif self.mode == 'combined':
            # 折中：α * quantile + (1-α) * ewma_R
            q = quantile if quantile is not None else 0.0
            e = self._ewma_R if self._ewma_R is not None else 0.0
            return self.combined_alpha * q + (1 - self.combined_alpha) * e
 
        else:
            raise ValueError(f"Unknown baseline mode: {self.mode}")
 
    def reset(self):
        """重置 EWMA 状态"""
        self._ewma_R = None
        self._ewma_R_e = None
        self._step_count = 0
 
    def state_dict(self):
        """导出状态（用于 checkpoint）"""
        return {
            'mode': self.mode,
            'ewma_R': self._ewma_R,
            'ewma_R_e': self._ewma_R_e,
            'step_count': self._step_count,
        }
 
    def load_state_dict(self, state):
        """加载状态"""
        self._ewma_R = state.get('ewma_R')
        self._ewma_R_e = state.get('ewma_R_e')
        self._step_count = state.get('step_count', 0)