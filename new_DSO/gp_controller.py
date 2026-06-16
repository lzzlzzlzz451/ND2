import numpy as np
import logging
from copy import deepcopy
from typing import List, Tuple, Optional
from .vocabulary import Vocabulary
from .program import Program
 
logger = logging.getLogger('new_DSO.GP')
 
 
class NDSubtreeInfo:
    """
    前缀表达式的子树信息提取器。
    利用 ND2 GDExpr 的 analysis_type / analysis_parent 来定位子树边界与类型。
    """
    def __init__(self, gdexpr, root_type: str):
        self.gdexpr = gdexpr
        self.root_type = root_type
 
    def get_subtree_spans(self, prefix: List[str]) -> List[Tuple[int, int, str]]:
        """
        返回所有子树的 (start, end, type) 信息。
 
        通过 analysis_type 标注每个位置的类型 (node/edge)，
        再根据前缀遍历规则计算子树边界。
 
        参数:
            prefix: token 名称列表
        返回:
            spans: [(start_idx, end_idx_exclusive, subtree_type), ...]
        """
        spans = []
        types = self.gdexpr.analysis_type(prefix, self.root_type)
        self._compute_spans(prefix, 0, types, spans)
        return spans
 
    def _compute_spans(self, prefix, start, types, spans):
        """递归计算子树边界"""
        if start >= len(prefix):
            return len(prefix)
        item = prefix[start]
        cur_type = types[start]
        pos = start + 1
 
        if item in self.gdexpr.operator.binary:
            pos = self._compute_spans(prefix, pos, types, spans)
            pos = self._compute_spans(prefix, pos, types, spans)
        elif item in self.gdexpr.operator.unary:
            pos = self._compute_spans(prefix, pos, types, spans)
        # else: terminal (arity=0), pos stays
 
        spans.append((start, pos, cur_type))
        return pos
 
    def extract_subtree(self, prefix: List[str], span_idx: int) -> Tuple[List[str], List[str], List[str], str]:
        """
        提取第 span_idx 个子树，返回。
 
        参数:
            prefix: 完整前缀表达式
            span_idx: 子树索引
        返回:
            left: 子树左边的部分
            subtree: 子树本身
            right: 子树右边的部分
            subtree_type: 'node' 或 'edge'
        """
        spans = self.get_subtree_spans(prefix)
        if span_idx < 0 or span_idx >= len(spans):
            raise IndexError(f"span_idx {span_idx} out of range [0, {len(spans)})")
        start, end, stype = spans[span_idx]
        left = prefix[:start]
        subtree = prefix[start:end]
        right = prefix[end:]
        return left, subtree, right, stype
 
 
class NDGPController:
    """
    网络动力学场景的 GP-Meld 遗传规划控制器。
 
    在 DSO 原版的 GP-Meld 基础上，增加 ND 图算子的类型约束：
    - 交叉：只能交换相同类型 (node/edge) 的子树
    - 变异：替换子树时必须保持类型一致
    - 图算子 (aggr/sour/targ) 的子树有严格的类型要求
 
    流程:
    1. 将 RNN 采样的 batch 作为种子种群
    2. 执行选择（锦标赛选择）
    3. 执行交叉（类型匹配的子树交换）
    4. 执行变异（decompose + random_fill_expr）
    5. 将 GP 产生的程序与 RL 采样合并
    """
    def __init__(self, vocab: Vocabulary, gdexpr, config,
                 reward_solver=None, root_type: str = 'node'):
        self.vocab = vocab
        self.gdexpr = gdexpr
        self.config = config
        self.reward_solver = reward_solver
        self.root_type = root_type
 
        gp_config = getattr(config, 'gp', None)
        if gp_config is None:
            gp_config = type('obj', (object,), {})()
 
        self.population_size = getattr(gp_config, 'population_size', 50)
        self.crossover_rate = getattr(gp_config, 'crossover_rate', 0.7)
        self.mutation_rate = getattr(gp_config, 'mutation_rate', 0.3)
        self.tournament_size = getattr(gp_config, 'tournament_size', 5)
        self.max_offspring = getattr(gp_config, 'max_offspring', 50)
        self.elite_fraction = getattr(gp_config, 'elite_fraction', 0.1)
        self.max_length = config.policy.max_length
 
        # 子树分析器
        self.subtree_info = NDSubtreeInfo(gdexpr, root_type)
 
    def evolve(self, programs: List[Program]) -> List[Program]:
        """
        第四步入口：对 RNN 采样的种群执行 GP 进化。
 
        参数:
            programs: RNN 采样得到的 Program 列表（种子种群）
        返回:
            gp_programs: GP 产生的新的 Program 列表
        """
        # 1. 筛选终端表达式作为种群
        population = [p for p in programs if p.is_terminal()]
        if len(population) < 2:
            logger.debug("[GP] 种群中终端表达式不足，跳过 GP")
            return []
 
        # 2. 按奖励排序
        population.sort(key=lambda p: p.reward, reverse=True)
 
        # 3. 保留精英
        n_elite = max(1, int(self.elite_fraction * len(population)))
        elites = population[:n_elite]
 
        # 4. 产生后代
        offspring = list(elites)  # 精英直接保留
        attempts = 0
        max_attempts = self.max_offspring * 3  # 防止无限循环
 
        while len(offspring) < self.population_size and attempts < max_attempts:
            attempts += 1
            r = np.random.random()
 
            if r < self.crossover_rate:
                # 交叉
                child = self._crossover(population)
            else:
                # 变异
                parent = self._tournament_select(population)
                child = self._mutate(parent)
 
            if child is not None:
                offspring.append(child)
 
        # 5. 去掉精英，只返回 GP 新产生的个体
        gp_programs = offspring[n_elite:]
        logger.info(f"[GP] 产生 {len(gp_programs)} 个新个体 "
                     f"(来自 {len(population)} 个种子)")
        return gp_programs
 
    # --------------------------------------------------
    # 锦标赛选择
    # --------------------------------------------------
    def _tournament_select(self, population: List[Program]) -> Program:
        """锦标赛选择：从随机 k 个个体中选奖励最高的"""
        k = min(self.tournament_size, len(population))
        candidates = np.random.choice(len(population), k, replace=False)
        best_idx = max(candidates, key=lambda i: population[i].reward)
        return population[best_idx]
 
    # --------------------------------------------------
    # 交叉（核心：类型匹配的子树交换）
    # --------------------------------------------------
    def _crossover(self, population: List[Program]) -> Optional[Program]:
        """
        ND 类型约束的子树交叉。
 
        步骤:
        1. 选择两个父代
        2. 分别提取子树信息 (span, type)
        3. 找到类型相同的子树对 (node↔node, edge↔edge)
        4. 交换子树，生成两个子代
        5. 验证合法性（长度、系数数量、类型一致性）
        """
        parent1 = self._tournament_select(population)
        parent2 = self._tournament_select(population)
        while parent2 is parent1 and len(population) > 1:
            parent2 = self._tournament_select(population)
 
        prefix1 = parent1.prefix
        prefix2 = parent2.prefix
 
        # 提取子树信息
        try:
            spans1 = self.subtree_info.get_subtree_spans(prefix1)
            spans2 = self.subtree_info.get_subtree_spans(prefix2)
        except Exception as e:
            logger.debug(f"[GP] 子树提取失败: {e}")
            return None
 
        if not spans1 or not spans2:
            return None
 
        # 按类型分组：node 类子树 vs edge 类子树
        type_groups_1 = {'node': [], 'edge': []}
        type_groups_2 = {'node': [], 'edge': []}
        for i, (s, e, t) in enumerate(spans1):
            if t in type_groups_1:
                type_groups_1[t].append(i)
        for i, (s, e, t) in enumerate(spans2):
            if t in type_groups_2:
                type_groups_2[t].append(i)
 
        # 找到两种类型都能匹配的
        common_types = [t for t in ['node', 'edge']
                        if type_groups_1[t] and type_groups_2[t]]
        if not common_types:
            logger.debug("[GP] 无类型匹配的子树对")
            return None
 
        # 随机选一个类型
        chosen_type = np.random.choice(common_types)
        idx1 = np.random.choice(type_groups_1[chosen_type])
        idx2 = np.random.choice(type_groups_2[chosen_type])
 
        # 提取子树
        left1, subtree1, right1, type1 = self.subtree_info.extract_subtree(prefix1, idx1)
        left2, subtree2, right2, type2 = self.subtree_info.extract_subtree(prefix2, idx2)
 
        # 交换生成子代
        child_prefix1 = left1 + subtree2 + right1
        child_prefix2 = left2 + subtree1 + right2
 
        # 选择较短的一个（避免膨胀）
        child_prefix = child_prefix1 if len(child_prefix1) <= len(child_prefix2) else child_prefix2
 
        # 验证
        if not self._validate_child(child_prefix):
            return None
 
        # 构建 Program
        return self._prefix_to_program(child_prefix)
 
    # --------------------------------------------------
    # 变异（decompose + random_fill_expr）
    # --------------------------------------------------
    def _mutate(self, parent: Program) -> Optional[Program]:
        """
        ND 类型约束的子树变异。
 
        策略（优先使用 GDExpr 的 decompose + random_fill_expr）:
        1. 用 decompose 将一个叶节点替换为 placeholder
        2. 用 random_fill_expr 在 placeholder 处生成新的子表达式
        3. random_fill_expr 内部会自动处理 aggr→edge, sour/targ→node 的类型切换
 
        备选策略: 直接随机替换一个子树
        """
        prefix = deepcopy(parent.prefix)
 
        # 策略 A: 使用 GDExpr 的 decompose（如果表达式有 placeholder 就不行，需要先填好）
        try:
            # decompose 会选择一个叶节点，替换为 placeholder，返回 (剩余prefix, 被移除的算子, 位置)
            remaining, removed_op, removed_idx = self.gdexpr.decompose(
                prefix, self.root_type, choose='random'
            )
 
            # 计算移除后空出的 token 数量
            old_subtree_len = len(prefix) - len(remaining) + 1  # +1 是 placeholder
            # 新子树的长度预算
            budget = min(old_subtree_len + np.random.randint(-1, 3),  # 允许小幅增减
                         self.max_length - len(remaining) + 1)  # 不超过最大长度
            budget = max(budget, 1)
 
            # 用 random_fill_expr 填充 placeholder
            child_prefix = self.gdexpr.random_fill_expr(
                total_len=len(remaining) - 1 + budget,  # 总长度
                prefix=remaining,
                var=60, coeff=30, const=10,  # 变量/系数/常量概率权重
                op_aggr=10, op_sour=10, op_targ=10,  # 图算子权重
            )
 
            if self._validate_child(child_prefix):
                return self._prefix_to_program(child_prefix)
 
        except Exception as e:
            logger.debug(f"[GP] decompose 变异失败: {e}")
 
        # 策略 B: 直接替换子树
        try:
            return self._subtree_mutation(prefix)
        except Exception as e:
            logger.debug(f"[GP] 子树变异失败: {e}")
            return None
 
    def _subtree_mutation(self, prefix: List[str]) -> Optional[Program]:
        """
        备选变异：直接选择一个子树并替换为随机生成的同类型子树。
        """
        spans = self.subtree_info.get_subtree_spans(prefix)
        if not spans:
            return None
 
        # 随机选一个非根子树（保留整体结构）
        candidates = [i for i, (s, e, t) in enumerate(spans) if s > 0]
        if not candidates:
            return None
 
        idx = np.random.choice(candidates)
        left, old_subtree, right, stype = self.subtree_info.extract_subtree(prefix, idx)
 
        # 生成同类型的新子树
        budget = min(len(old_subtree) + np.random.randint(-1, 2),
                     self.max_length - len(left) - len(right))
        budget = max(budget, 1)
 
        # 构造带 placeholder 的骨架，然后用 random_fill_expr 填充
        skeleton = left + [stype] + right
        try:
            child_prefix = self.gdexpr.random_fill_expr(
                total_len=len(left) + budget + len(right),
                prefix=skeleton,
                var=60, coeff=30, const=10,
                op_aggr=10, op_sour=10, op_targ=10,
            )
        except Exception:
            return None
 
        if self._validate_child(child_prefix):
            return self._prefix_to_program(child_prefix)
        return None
 
    # --------------------------------------------------
    # 辅助方法
    # --------------------------------------------------
    def _validate_child(self, prefix: List[str]) -> bool:
        """验证子代表达式的合法性"""
        # 长度检查
        if len(prefix) > self.max_length:
            return False
        if len(prefix) == 0:
            return False
 
        # 系数数量检查
        max_coeff = self.config.policy.max_coeff_num
        if prefix.count('<C>') > max_coeff:
            return False
 
        # 不能有未填充的 placeholder
        if 'node' in prefix or 'edge' in prefix:
            return False
 
        # 类型一致性验证：尝试 analysis_type
        try:
            self.gdexpr.analysis_type(prefix, self.root_type)
        except (AssertionError, ValueError):
            return False
 
        return True
 
    def _prefix_to_program(self, prefix: List[str]) -> Optional[Program]:
        """将 prefix 字符串列表转换为 Program 对象"""
        # 将 token 名称转为 id
        token_ids = []
        for token in prefix:
            if token in self.vocab.word2id:
                token_ids.append(self.vocab.word2id[token])
            else:
                # 数值型系数（来自 BFGS 拟合后的替换）
                try:
                    float(token)
                    # 用 <C> 占位代替（变异后的新表达式需要重新拟合系数）
                    token_ids.append(self.vocab.word2id['<C>'])
                except (ValueError, TypeError):
                    logger.debug(f"[GP] 未知 token: {token}")
                    return None
 
        # 查缓存
        cache_key = tuple(token_ids)
        if cache_key in Program._cache:
            return Program._cache[cache_key]
 
        # 新建 Program（会自动补全）
        prog = Program(
            token_ids=token_ids,
            vocab=self.vocab,
            config=self.config,
            gdexpr=self.gdexpr,
            reward_solver=self.reward_solver,
        )
        Program._cache[cache_key] = prog
        return prog