import torch
import numpy as np
from .vocabulary import Vocabulary   # ← 加这一行
 
class ValidMaskComputer:
    """
    计算每一步的合法 action 掩码，确保采样的前缀表达式：
    1. 长度不超限
    2. 系数数量不超限
    3. ND 图算子类型约束（aggr 需要边输入，sour/targ 需要节点输入）
    4. placeholder 必须被正确填入
    """
    def __init__(self, vocab: Vocabulary, config, available_vars=None):
        self.vocab = vocab
        self.max_length = config.policy.max_length
        self.max_coeff_num = config.policy.max_coeff_num
        self.max_node_coeff_num = config.policy.max_node_coeff_num
        self.max_edge_coeff_num = config.policy.max_edge_coeff_num
        self.root_type = config.data.root_type
 
        # 预计算各类 token id 集合
        self._placeholder_ids = set(self.vocab.tokens_of_kind('placeholder'))
        self._coefficient_ids = set(self.vocab.tokens_of_kind('coefficient'))
        self._node_coeff_ids = set(self.vocab.tokens_of_kind('node_coeff'))
        self._edge_coeff_ids = set(self.vocab.tokens_of_kind('edge_coeff'))
        self._variable_node_ids = set(self.vocab.tokens_of_kind('variable'))
        self._variable_edge_ids = set(
            tid for tid, k in self.vocab._kind.items()
            if k == 'variable' and self.vocab._type_scope[tid] == 'edge'
        )
        self._variable_node_ids -= self._variable_edge_ids
        self._binary_ids = set(self.vocab.tokens_of_kind('operator'))
        self._binary_ids = set(tid for tid in self._binary_ids if self.vocab.arity(tid) == 2)
        self._unary_ids = set(tid for tid in self.vocab.tokens_of_kind('operator')
                              if self.vocab.arity(tid) == 1)
        self._max_id = max(vocab.word2id.values())
        self._n_actions = self._max_id + 1
        
        if available_vars is not None:
            avail_set = set(available_vars)
            self._variable_node_ids = set(
                tid for token, tid in vocab.word2id.items()
                if vocab.kind(tid) == 'variable' and vocab.type_scope(tid) == 'node'
                and token in avail_set
            )
            self._variable_edge_ids = set(
                tid for token, tid in vocab.word2id.items()
                if vocab.kind(tid) == 'variable' and vocab.type_scope(tid) == 'edge'
                and token in avail_set
            )
 
    def compute_mask(self, prefix_token_ids, dangling, coeff_counts, has_variable=False):
        n_words = self.vocab.n_words
        mask = np.zeros(self._max_id + 1, dtype=bool)
        remaining = self.max_length - len(prefix_token_ids)
        if remaining <= 0 or dangling <= 0:
            return mask
    
        # 追踪当前待填充位置的类型
        current_type = self._get_current_type(prefix_token_ids)
    
        # ★ 关键：第一步和表达式太短时，强制选算子，不允许直接出变量结束
        is_first_token = len(prefix_token_ids) == 0
        too_short = len(prefix_token_ids) < 2  # 至少2个token才算合法表达式
    
        # ---- 算子总是开放（受长度和类型约束） ----
        # 二元算子：需要 remaining >= 3（算子 + 2个参数各至少1 token）
        if remaining >= 3:
            mask[list(self._binary_ids)] = True
        # 一元算子：需要 remaining >= 2
        if remaining >= 2:
            mask[list(self._unary_ids)] = True
    
        # ---- 类型约束：禁止类型不匹配的图算子 ----
        if current_type == 'edge':
            for tid in self._unary_ids:
                token = self.vocab.id2word.get(tid, '')
                if token in ('aggr', 'rgga'):
                    mask[tid] = False
            # edge + dangling==1：二元算子会增加 dangling，不允许
            if dangling == 1:
                mask[list(self._binary_ids)] = False
        elif current_type == 'node':
            for tid in self._unary_ids:
                token = self.vocab.id2word.get(tid, '')
                if token in ('sour', 'targ'):
                    mask[tid] = False
    
        # ---- 强制至少使用一个变量（dangling==1 且无变量时） ----
        if dangling == 1 and not has_variable:
            # 禁止直接结束，必须继续生长
            # 关掉所有终止符（变量、常量、系数）
            mask[list(self._variable_node_ids)] = False
            mask[list(self._variable_edge_ids)] = False
            mask[list(self.vocab.tokens_of_kind('constant'))] = False
            mask[self.vocab.word2id.get('<C>', -1)] = False
            mask[self.vocab.word2id.get('<Cv>', -1)] = False
            mask[self.vocab.word2id.get('<Ce>', -1)] = False
            # 只保留算子（上面已经开了，类型约束也处理了）
            return mask
    
        # ---- ★ 终止符：第一步和太短时不开放 ----
        allow_terminators = not is_first_token and not too_short
    
        if remaining >= 1 and allow_terminators:
            # 变量
            if current_type == 'node':
                mask[list(self._variable_node_ids)] = True
            elif current_type == 'edge':
                mask[list(self._variable_edge_ids)] = True
            else:
                mask[list(self._variable_node_ids)] = True
                mask[list(self._variable_edge_ids)] = True
    
            # 常量/系数：子树中必须有变量才允许
            constant_ok = has_variable or self._check_constant_ok(prefix_token_ids)
            if not constant_ok:
                constant_ok = dangling > 1 or len(prefix_token_ids) > 0
            if constant_ok:
                if current_type != 'edge':
                    mask[list(self.vocab.tokens_of_kind('constant'))] = True
                if coeff_counts[0] < self.max_coeff_num and current_type != 'edge':
                    mask[self.vocab.word2id.get('<C>', -1)] = True
                # if coeff_counts[1] < self.max_node_coeff_num and current_type in ('node', 'scalar', 'any'):
                #     mask[self.vocab.word2id.get('<Cv>', -1)] = True
    
        return mask
    
    
    def _get_current_type(self, prefix_token_ids):
        """追踪当前待填充位置期望的类型，和 ND2 的 act 逻辑一致"""
        type_stack = [self.root_type]  # 根类型
        
        for tid in prefix_token_ids:
            if not type_stack:
                break
            expected = type_stack.pop()
            token = self.vocab.id2word.get(tid, '')
            
            # 确定 token 的输出类型
            if token == 'aggr':
                output_type = 'node'
            elif token in ('sour', 'targ'):
                output_type = 'edge'
            elif self.vocab.kind(tid) == 'variable':
                output_type = self.vocab.type_scope(tid)  # 'node' or 'edge'
            elif token in self.vocab.tokens_of_kind('constant') or token in ('<C>',):
                output_type = 'scalar'
            elif token in ('<Cv>',):
                output_type = 'node'
            elif token in ('<Ce>',):
                output_type = 'edge'
            else:
                # 透传型算子（一元/二元）：输出类型 = 期望类型
                output_type = expected
            
            # 根据 arity 压入参数类型
            arity = self.vocab.arity(tid)
            if arity == 2:
                if token == 'regular':
                    type_stack.append(output_type)  # 左参数
                    type_stack.append(output_type)  # 右参数
                else:
                    type_stack.append(output_type)
                    type_stack.append(output_type)
            elif arity == 1:
                if token == 'aggr':
                    type_stack.append('edge')   # aggr 期望 edge 输入
                elif token in ('sour', 'targ'):
                    type_stack.append('node')   # sour/targ 期望 node 输入
                else:
                    type_stack.append(expected) # 其他一元算子透传
            # arity == 0: 变量/常量，不压栈
        
        return type_stack[-1] if type_stack else self.root_type
    
    
    def _check_constant_ok(self, prefix_token_ids):
        """检查当前子树中是否已有变量（和 ND2 的 constant_ok 逻辑一致）"""
        # 简化版：扫描前缀中是否有变量 token
        for tid in prefix_token_ids:
            if self.vocab.kind(tid) == 'variable':
                return True
        return False
 
    def _infer_expected_type(self, prefix_token_ids):
        """
        从左到右遍历前缀，追踪每个位置的实际类型，
        返回当前 dangling 位置期望的类型。
        """
        # 类型栈：记录每个未填充 slot 期望的类型
        # 初始：根位置期望 root_type
        type_stack = [self.root_type]
        
        for tid in prefix_token_ids:
            if not type_stack:
                break
            # 弹出当前 token 填充的 slot 的期望类型
            expected = type_stack.pop()
            token = self.vocab.id2word[tid]
            
            # 确定 token 的输出类型
            if token in ('aggr',):
                output_type = 'node'
            elif token in ('sour', 'targ'):
                output_type = 'edge'
            elif token in self.vocab.word2id and self.vocab.kind(tid) == 'variable':
                output_type = self.vocab.type_scope(tid)
            elif token in self.vocab.word2id and self.vocab.kind(tid) == 'constant':
                output_type = 'scalar'
            elif token in ('neg', 'abs', 'inv', 'exp', 'logabs', 'sin', 'cos', 'tan',
                        'sqrtabs', 'pow2', 'pow3', 'tanh', 'sigmoid'):
                # 一元算子：输出类型 = 输入类型（传递型）
                output_type = expected  # 透传期望类型
            elif token in ('add', 'sub', 'mul', 'div'):
                # 二元算子：输出类型 = 左操作数类型
                output_type = expected
            elif token == 'pow':
                output_type = expected
            elif token == 'regular':
                output_type = expected
            else:
                output_type = expected
            
            # 根据 arity 压入新 slot
            arity = self.vocab.arity(tid)
            if arity == 2:
                type_stack.append(output_type)  # 左参数
                type_stack.append(output_type)  # 右参数
            elif arity == 1:
                # 一元算子参数类型
                if token == 'aggr':
                    type_stack.append('edge')   # aggr 期望 edge 输入
                elif token in ('sour', 'targ'):
                    type_stack.append('node')   # sour/targ 期望 node 输入
                else:
                    type_stack.append(expected) # 其他一元算子透传
        
        # 栈顶就是当前 dangling 位置期望的类型
        return type_stack[-1] if type_stack else self.root_type
 
    def compute_mask_batch(self, prefixes, danglings, coeff_counts_list, has_variables=None):
        B = len(prefixes)
        masks = np.zeros((B, self._n_actions), dtype=bool)
        for i in range(B):
            hv = has_variables[i] if has_variables is not None else False
            masks[i] = self.compute_mask(prefixes[i], danglings[i], coeff_counts_list[i], has_variable=hv)
        return masks