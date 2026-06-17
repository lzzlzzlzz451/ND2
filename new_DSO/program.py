# ============================================================
# new_DSO/program.py — 更新 Program 类（增量修改）
# ============================================================
# 只展示变更部分，其余与第一二步一致
 
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from ND2.GDExpr import GDExpr
 
logger = logging.getLogger('new_DSO.Program')
 
class Program:
    _cache = {}
 
    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
 
    # 奖励缓存（跨所有 Program 实例共享）
    _reward_cache = {}
 
    @classmethod
    def clear_reward_cache(cls):
        cls._reward_cache.clear()
 
    def __init__(self, token_ids, vocab, config, gdexpr, reward_solver=None):
        self.vocab = vocab
        self.config = config
        self.gdexpr = gdexpr
        self.reward_solver = reward_solver
 
        # 第二步：补全表达式（同前，省略）
        self.token_ids = self._complete_expression(token_ids)
        # self.prefix = [vocab.id2word[tid] for tid in self.token_ids]
        self.prefix = [vocab.id2word[tid] for tid in self.token_ids if tid != vocab.pad_id]
 
        # 惰性奖励
        self._reward = None
        self._prefix_with_coef = None
        self._metrics = None
 
    # -------------------------------------------------------
    # 第三步核心：计算奖励
    # -------------------------------------------------------
    @property
    def reward(self):
        """惰性奖励属性：首次访问时触发完整的 BFGS 参数拟合 + MSE 评估"""
        if self._reward is None:
            self._compute_reward()
        return self._reward if self._reward is not None else -np.inf
 
    @property
    def prefix_with_coef(self):
        """系数被具体数值替换后的 prefix"""
        if self._prefix_with_coef is None:
            self._compute_reward()
        return self._prefix_with_coef or self.prefix
 
    @property
    def metrics(self):
        """完整精度评估指标"""
        if self._metrics is None:
            self._compute_full_metrics()
        return self._metrics
 
    def _compute_reward(self):
        if not self.is_terminal():
            self._reward = 0.0; return
        if self.reward_solver is None:
            self._reward = -np.inf; return
        try:
            reward, pwc = self.reward_solver.solve(self.prefix, sample=True, max_iter=30)
        except Exception as e:
            self._reward = 0.0; return
        self._reward = reward; self._prefix_with_coef = pwc
 
    def _compute_full_metrics(self):
        """
        用全量数据 + 多次迭代做精确评估（仅在表达式表现好时调用）。
        """
        if self.reward_solver is None or self.prefix_with_coef is None:
            self._metrics = {}
            return
 
        try:
            self._metrics = self.reward_solver.evaluate(self.prefix_with_coef, {})
        except Exception:
            self._metrics = {}
 
    def recompute_reward_precise(self):
        if not self.is_terminal() or self.reward_solver is None: return self.reward
        try:
            reward, pwc = self.reward_solver.solve(self.prefix, sample=False, max_iter=1000)
            self._reward = reward; self._prefix_with_coef = pwc
        except: pass
        return self._reward
 
    # 其余方法同第一二步，不再重复
    def _complete_expression(self, token_ids):
        dangling = 1          # 需要1个根表达式
        cut_idx = len(token_ids)
        for i, tid in enumerate(token_ids):
            dangling = dangling - 1 + self.vocab.arity(tid)
            if dangling <= 0:
                cut_idx = i + 1
                break
        result = list(token_ids[:cut_idx])
        if dangling > 0:
            default_id = self.vocab.word2id.get(
                'v1' if self.config.data.root_type == 'node' else 'e1')
            if default_id is not None:
                while dangling > 0:
                    result.append(default_id)
                    dangling -= 1
        return result
 
    def is_terminal(self):
        for tid in self.token_ids:
            if self.vocab.kind(tid) == 'placeholder':
                return False
        return True
 
    def __repr__(self):
        return f"Program({' '.join(self.prefix)})"
 
    def __hash__(self):
        return hash(tuple(self.token_ids))
 
    def __eq__(self, other):
        return isinstance(other, Program) and self.token_ids == other.token_ids