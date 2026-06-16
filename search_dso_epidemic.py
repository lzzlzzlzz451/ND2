# ============================================================
# new_DSO/config.py — 配置系统
# ============================================================
from ND2.utils import AttrDict
 
def get_default_config():
    """返回 new_DSO 的默认配置，兼容 ND2 的图动力学词汇表"""
    config = AttrDict({
        # ---- 词汇表 ----
        'vocabulary': AttrDict({
            'special': AttrDict({'pad': 0, 'sos': 1, 'eos': 2}),
            'placeholder': AttrDict({'node': 3, 'edge': 4}),
            'coefficient': 5,         # <C> 标量系数
            'node_coefficient': 6,    # <Cv> 节点级系数
            'edge_coefficient': 7,    # <Ce> 边级系数
            'variable': AttrDict({
                'node': AttrDict({'v1': 10, 'v2': 11, 'v3': 12, 'v4': 13, 'v5': 14}),
                'edge': AttrDict({'e1': 15, 'e2': 16, 'e3': 17, 'e4': 18, 'e5': 19}),
            }),
            'constant': AttrDict({
                '1': 21, '2': 22, '3': 23, '4': 24, '5': 25,
                '(1/2)': 26, '(1/3)': 27, '(1/4)': 28, '(1/5)': 29,
            }),
            'operator': AttrDict({
                'binary': AttrDict({
                    'add': 31, 'sub': 32, 'mul': 33, 'div': 34,
                    'pow': 35, 'regular': 37,
                }),
                'unary': AttrDict({
                    'neg': 38, 'exp': 39, 'logabs': 40,
                    'sin': 41, 'cos': 42, 'tan': 43,
                    'abs': 44, 'inv': 45, 'sqrtabs': 46,
                    'pow2': 47, 'pow3': 48, 'tanh': 51,
                    'sigmoid': 52,
                    # --- ND 图算子 ---
                    'aggr': 53, 'sour': 54, 'targ': 55,
                }),
            }),
        }),
 
        # ---- RNN 策略网络 ----
        'policy': AttrDict({
            'hidden_size': 64,
            'num_layers': 2,
            'cell_type': 'lstm',       # 'lstm' 或 'gru'
            'embedding_size': 32,
            'max_length': 30,          # 表达式最大 token 数
            'max_coeff_num': 5,        # 最大标量系数数量
            'max_node_coeff_num': 3,   # 最大节点系数数量
            'max_edge_coeff_num': 3,   # 最大边系数数量
        }),
 
        # ---- 训练 ----
        'training': AttrDict({
            'batch_size': 256,
            'n_samples': 100000,
            'epsilon': 0.05,           # 风险寻求：只保留 top-5%
            'entropy_weight': 0.01,
            'baseline_mode': 'R_e',    # 'R_e', 'ewma_R', 'ewma_R_e', 'combined'
            'learning_rate': 1e-3,
            'seed': 42,
        }),
 
        # ---- 数据 ----
        'data': AttrDict({
            'root_type': 'node',       # 'node' 或 'edge'
            'sample_num': 500,         # BFGS 采样点数
            'complexity_base': 0.999,
        }),
    })
    return config
 
 
# ============================================================
# new_DSO/vocabulary.py — 词汇表 + 算子元信息
# ============================================================
import numpy as np
from ND2.utils import AttrDict
 
class Vocabulary:
    """
    统一管理 ND2 风格的 token 词汇表，并提供每个 token 的元信息：
    - arity: 操作数数量（0=终端, 1=一元, 2=二元）
    - kind: 'special'/'placeholder'/'coefficient'/'node_coeff'/'edge_coeff'
            /'variable'/'constant'/'operator'
    - type_scope: 该 token 期望的输入类型 ('node'/'edge'/'any')
    """
    def __init__(self, config):
        self.config = config
        voc = config.vocabulary
 
        # 合并所有 word2id
        self.word2id = (voc.special + voc.placeholder +
                        AttrDict({'<C>': voc.coefficient,
                                  '<Cv>': voc.node_coefficient,
                                  '<Ce>': voc.edge_coefficient}) +
                        voc.constant + voc.variable.node + voc.variable.edge +
                        voc.operator.binary + voc.operator.unary)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.n_words = len(self.word2id)
        self.pad_id = voc.special.pad
        self.sos_id = voc.special.sos
        self.eos_id = voc.special.eos
 
        # 预计算每个 token 的 arity 和类型信息
        self._arity = {}
        self._kind = {}
        self._type_scope = {}  # 输出类型: 'node' / 'edge' / 'scalar'
 
        for token, tid in self.word2id.items():
            if token in voc.special:
                self._arity[tid] = 0
                self._kind[tid] = 'special'
                self._type_scope[tid] = 'any'
            elif token in voc.placeholder:
                self._arity[tid] = 0
                self._kind[tid] = 'placeholder'
                self._type_scope[tid] = token  # 'node' 或 'edge'
            elif token == '<C>':
                self._arity[tid] = 0
                self._kind[tid] = 'coefficient'
                self._type_scope[tid] = 'scalar'
            elif token == '<Cv>':
                self._arity[tid] = 0
                self._kind[tid] = 'node_coeff'
                self._type_scope[tid] = 'node'
            elif token == '<Ce>':
                self._arity[tid] = 0
                self._kind[tid] = 'edge_coeff'
                self._type_scope[tid] = 'edge'
            elif token in voc.variable.node:
                self._arity[tid] = 0
                self._kind[tid] = 'variable'
                self._type_scope[tid] = 'node'
            elif token in voc.variable.edge:
                self._arity[tid] = 0
                self._kind[tid] = 'variable'
                self._type_scope[tid] = 'edge'
            elif token in voc.constant:
                self._arity[tid] = 0
                self._kind[tid] = 'constant'
                self._type_scope[tid] = 'scalar'
            elif token in voc.operator.binary:
                self._arity[tid] = 2
                self._kind[tid] = 'operator'
                self._type_scope[tid] = self._infer_binary_type(token)
            elif token in voc.operator.unary:
                self._arity[tid] = 1
                self._kind[tid] = 'operator'
                self._type_scope[tid] = self._infer_unary_type(token)
 
        # 构建 arity 数组，便于向量化计算 dangling
        self.arity_array = np.array([self._arity[i] for i in range(self.n_words)],
                                    dtype=np.int32)
 
    def _infer_unary_type(self, token):
        """一元算子的输出类型推断"""
        # 图算子会改变维度
        type_map = {
            'aggr': 'node',    # edge -> node (聚合边到目标节点)
            'sour': 'edge',    # node -> edge (取源节点值)
            'targ': 'edge',    # node -> edge (取目标节点值)
        }
        return type_map.get(token, 'any')  # 其余一元算子保持输入类型
 
    def _infer_binary_type(self, token):
        """二元算子的输出类型推断"""
        return 'any'  # 二元算子类型取决于操作数
 
    def arity(self, token_id):
        return self._arity.get(token_id, 0)
 
    def kind(self, token_id):
        return self._kind.get(token_id, 'unknown')
 
    def type_scope(self, token_id):
        return self._type_scope.get(token_id, 'any')
 
    def tokens_of_kind(self, kind):
        """返回所有指定 kind 的 token id"""
        return [tid for tid, k in self._kind.items() if k == kind]
 
    def is_terminal(self, token_id):
        """token 是否为终端（arity=0 且不是 placeholder）"""
        return self._arity[token_id] == 0 and self._kind[token_id] != 'placeholder'
 
 
# ============================================================
# new_DSO/valid_mask.py — ND 图约束的有效掩码
# ============================================================
import torch
import numpy as np
 
class ValidMaskComputer:
    """
    计算每一步的合法 action 掩码，确保采样的前缀表达式：
    1. 长度不超限
    2. 系数数量不超限
    3. ND 图算子类型约束（aggr 需要边输入，sour/targ 需要节点输入）
    4. placeholder 必须被正确填入
    """
    def __init__(self, vocab: Vocabulary, config):
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
 
    def compute_mask(self, prefix_token_ids, dangling, coeff_counts):
        """
        参数:
            prefix_token_ids: list[int], 当前已采样的 token id 序列
            dangling: int, 当前悬空节点数（还需填入的参数槽位）
            coeff_counts: tuple(int,int,int), (<C>, <Cv>, <Ce>) 已用数量
 
        返回:
            mask: np.ndarray, shape=(n_words,), bool, True=合法
        """
        n_words = self.vocab.n_words
        mask = np.zeros(n_words, dtype=bool)
        remaining = self.max_length - len(prefix_token_ids)
 
        if remaining <= 0:
            return mask  # 不允许再采样
 
        # --- 悬空 > 0 时可以继续填 ---
        if dangling > 0:
            # 二元算子: 至少需要 2 个空位（算子本身 + 2个子节点 = 净增 1 个悬空）
            if remaining >= 2:
                mask[list(self._binary_ids)] = True
            # 一元算子: 至少需要 1 个空位（净减 0 个悬空）
            if remaining >= 1:
                mask[list(self._unary_ids)] = True
 
            # 终端: 需要 >= 1 个空位且能减少悬空
            if remaining >= 1:
                # 变量: 根据当前期望的 placeholder 类型选择
                expected_type = self._infer_expected_type(prefix_token_ids)
                if expected_type == 'node':
                    mask[list(self._variable_node_ids)] = True
                elif expected_type == 'edge':
                    mask[list(self._variable_edge_ids)] = True
                else:
                    # 'any': 两种都可以
                    mask[list(self._variable_node_ids)] = True
                    mask[list(self._variable_edge_ids)] = True
 
                # 常量
                mask[list(self.vocab.tokens_of_kind('constant'))] = True
 
                # 标量系数 <C>
                if coeff_counts[0] < self.max_coeff_num:
                    mask[self.vocab.word2id['<C>']] = True
                # 节点系数 <Cv>
                if coeff_counts[1] < self.max_node_coeff_num and expected_type != 'edge':
                    mask[self.vocab.word2id['<Cv>']] = True
                # 边系数 <Ce>
                if coeff_counts[2] < self.max_edge_coeff_num and expected_type != 'node':
                    mask[self.vocab.word2id['<Ce>']] = True
 
        # --- 图算子特殊约束 ---
        # aggr(一元, edge->node): 子表达式必须是 edge 类型
        # sour(一元, node->edge): 子表达式必须是 node 类型
        # targ(一元, node->edge): 子表达式必须是 node 类型
        # 这些约束通过 expected_type 间接实现
 
        # --- placeholder ---
        # 在 DSO 中不用 placeholder，直接由变量/系数填入
        # 但保留兼容性：如果 dangling==0 且表达式未结束，不需要 placeholder
 
        return mask
 
    def _infer_expected_type(self, prefix_token_ids):
        """
        推断当前悬空位置期望填入的类型（'node'/'edge'/'any'）
        通过回溯最近的算子来判断
        """
        # 简化版本：追踪最近的图算子上下文
        # 例如 aggr 后面期望 edge 类型输入
        # sour/targ 后面期望 node 类型输入
        for tid in reversed(prefix_token_ids):
            token = self.vocab.id2word[tid]
            if token == 'aggr':
                return 'edge'  # aggr 期望边变量作为输入
            elif token in ('sour', 'targ'):
                return 'node'  # sour/targ 期望节点变量作为输入
        return self.root_type  # 默认由根类型决定
 
    def compute_mask_batch(self, prefixes, danglings, coeff_counts_list):
        """批量计算掩码"""
        B = len(prefixes)
        masks = np.zeros((B, self.vocab.n_words), dtype=bool)
        for i in range(B):
            masks[i] = self.compute_mask(prefixes[i], danglings[i], coeff_counts_list[i])
        return masks
 
 
# ============================================================
# new_DSO/policy.py — RNN 策略网络 (PyTorch)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
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
        self.hidden_size = pc.hidden_size
        self.num_layers = pc.num_layers
        self.max_length = pc.max_length
        self.embedding_size = pc.embedding_size
 
        # Token embedding
        self.token_embedding = nn.Embedding(self.n_words, self.embedding_size,
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
        self.output_head = nn.Linear(self.hidden_size, self.n_words)
 
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
 
                valid_masks = valid_mask_computer.compute_mask_batch(
                    prefix_lists, dangling_list, coeff_list
                )  # (B, n_words), numpy bool
 
                # 已完成的样本掩码全 False
                for i in range(batch_size):
                    if finished[i]:
                        valid_masks[i, :] = False
 
                valid_masks_t = torch.from_numpy(valid_masks).to(device)
 
                # 将不合法 token 的 logits 设为 -inf
                logits = logits.masked_fill(~valid_masks_t, float('-inf'))
 
                # 计算概率分布
                probs = F.softmax(logits, dim=-1)  # (B, n_words)
 
                # 采样
                dist = torch.distributions.Categorical(probs)
                sampled = dist.sample()  # (B,)
 
                # 记录
                for i in range(batch_size):
                    if not finished[i]:
                        actions_batch[i, t] = sampled[i]
                        log_probs_batch[i, t] = dist.log_prob(sampled[i])
                        entropies_batch[i, t] = dist.entropy()
                        masks_batch[i, t] = True
 
                        # 更新 dangling
                        tid = sampled[i].item()
                        arity = self.vocab.arity(tid)
                        danglings[i] = danglings[i] - 1 + arity  # 填一个槽位，开 arity 个新槽
 
                        # 更新系数计数
                        if self.vocab.kind(tid) == 'coefficient':
                            coeff_c[i] += 1
                        elif self.vocab.kind(tid) == 'node_coeff':
                            coeff_cv[i] += 1
                        elif self.vocab.kind(tid) == 'edge_coeff':
                            coeff_ce[i] += 1
 
                        # 检查是否完成（悬空=0）
                        if danglings[i] <= 0:
                            finished[i] = True
 
                current_actions = sampled
 
            return (actions_batch, log_probs_batch, entropies_batch,
                    finished, masks_batch)
 
 
# ============================================================
# new_DSO/program.py — 表达式对象 (Program)
# ============================================================
import numpy as np
from copy import deepcopy
from ND2.GDExpr import GDExprClass
 
class Program:
    """
    一个完整的前缀表达式对象，对应 DSO 中的 Program。
 
    封装了：
    - 前缀 token 序列
    - 表达式补全（dangling 归零 / 默认填充）
    - 缓存机制
    - 惰性奖励计算（延迟到第一次访问 p.reward 时触发）
    """
    # 类级缓存
    _cache = {}
 
    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
 
    def __init__(self, token_ids, vocab: Vocabulary, config, gdexpr: GDExprClass,
                 reward_solver=None):
        """
        参数:
            token_ids: list[int], 采样的 token id 序列
            vocab: Vocabulary
            config: AttrDict
            gdexpr: GDExprClass, 用于表达式求值和参数拟合
            reward_solver: RewardSolver, 用于计算奖励（惰性）
        """
        self.vocab = vocab
        self.config = config
        self.gdexpr = gdexpr
        self.reward_solver = reward_solver
 
        # ---- 第二步：补全表达式 ----
        self.token_ids = self._complete_expression(token_ids)
        self.prefix = [vocab.id2word[tid] for tid in self.token_ids]
 
        # 惰性奖励
        self._reward = None
        self._coef_dict = None
 
    def _complete_expression(self, token_ids):
        """
        第二步核心：补全不完整的前缀表达式。
 
        逻辑:
        1. 追踪 dangling = 1 + cumsum(arities - 1)
        2. 若 dangling 中途归零 → 截断，表达式已完成
        3. 若遍历完 dangling > 0 → 用默认变量（如 v1/e1）填充
        """
        dangling = 0
        cut_idx = len(token_ids)
 
        for i, tid in enumerate(token_ids):
            if i == 0:
                dangling = 1  # 根节点占 1 个槽位
            else:
                dangling -= 1  # 填入一个槽位
 
            arity = self.vocab.arity(tid)
            dangling += arity  # 该 token 开辟 arity 个子槽位
 
            if dangling <= 0:
                cut_idx = i + 1
                break
 
        result = list(token_ids[:cut_idx])
 
        # 如果仍有悬空节点，用默认变量填充
        if dangling > 0:
            root_type = self.config.data.root_type
            if root_type == 'node':
                default_var_id = self.vocab.word2id.get('v1', None)
            else:
                default_var_id = self.vocab.word2id.get('e1', None)
 
            if default_var_id is not None:
                while dangling > 0:
                    result.append(default_var_id)
                    dangling -= 1  # 变量 arity=0，填一个槽位
 
        return result
 
    @property
    def reward(self):
        """惰性奖励属性"""
        if self._reward is None and self.reward_solver is not None:
            self._compute_reward()
        return self._reward if self._reward is not None else -np.inf
 
    @property
    def coef_dict(self):
        """惰性系数属性"""
        if self._coef_dict is None and self.reward_solver is not None:
            self._compute_reward()
        return self._coef_dict
 
    def _compute_reward(self):
        """
        计算奖励（第三步的前置接口，此处仅做参数拟合和 MSE 评估）。
        完整的奖励计算将在第三步实现。
        """
        # 占位：第三步会完整实现
        # 目前先标记为待计算
        self._reward = -np.inf
        self._coef_dict = {}
 
    def is_terminal(self):
        """表达式是否完整（无未填充的 placeholder）"""
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
 
 
# ============================================================
# new_DSO/core.py — DSO 核心训练循环 (第零步初始化 + 采样骨架)
# ============================================================
import os
import torch
import numpy as np
import logging
from .config import get_default_config
from .vocabulary import Vocabulary
from .valid_mask import ValidMaskComputer
from .policy import RNVPolicy
from .program import Program
from ND2.GDExpr import GDExprClass
from ND2.utils import AttrDict, seed_all
 
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
 
        # 0.8 训练状态
        self.n_evals = 0
        self.best_reward = -np.inf
        self.best_program = None
 
        # 清空 Program 缓存
        Program.clear_cache()
 
        logger.info("[new_DSO] 第零步初始化完成。")
 
    def _merge_gdexpr_config(self, dso_config, gdexpr_config):
        """将 GDExpr 所需的配置合并到 DSO 配置中"""
        merged = AttrDict(dict(dso_config))
        for key in ['max_complexity', 'max_coeff_num']:
            if key in gdexpr_config:
                merged.policy[key] = gdexpr_config[key]
        return merged
 
    def sample_batch(self, batch_size=None):
        """
        第一步 + 第二步：采样一个 batch 的表达式并构建 Program 对象。
 
        第一步: RNN 自回归采样 token 序列
        第二步: 补全不完整表达式 → 构建 Program
 
        返回:
            programs: list[Program]
            actions: (B, L) tensor
            log_probs: (B, L) tensor
            entropies: (B, L) tensor
            masks: (B, L) tensor
        """
        batch_size = batch_size or self.config.training.batch_size
 
        # ---- 第一步：RNN 自回归采样 ----
        actions, log_probs, entropies, finished, masks = \
            self.policy.sample(
                batch_size=batch_size,
                valid_mask_computer=self.valid_mask_computer,
                device=self.device,
                prior_bias=self._get_prior_bias()
            )
 
        # ---- 第二步：补全与构建 Program ----
        programs = []
        for i in range(batch_size):
            # 提取有效 token（去除 padding）
            valid_mask_i = masks[i]
            token_ids = actions[i][valid_mask_i].cpu().tolist()
 
            # 查缓存
            cache_key = tuple(token_ids)
            if cache_key in Program._cache:
                programs.append(Program._cache[cache_key])
            else:
                prog = Program(
                    token_ids=token_ids,
                    vocab=self.vocab,
                    config=self.config,
                    gdexpr=self.gdexpr,
                    reward_solver=None,  # 第三步再绑定
                )
                Program._cache[cache_key] = prog
                programs.append(prog)
 
        self.n_evals += batch_size
 
        return programs, actions, log_probs, entropies, masks
 
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
        """
        执行一次完整的 DSO 迭代（第零步到第二步）。
        第三步到第八步将在后续实现后补全。
        """
        programs, actions, log_probs, entropies, masks = self.sample_batch()
 
        # 输出当前 batch 的统计信息
        terminal_count = sum(1 for p in programs if p.is_terminal())
        avg_length = np.mean([len(p.token_ids) for p in programs])
        logger.info(
            f"[new_DSO] 采样 {len(programs)} 个表达式, "
            f"完整: {terminal_count}/{len(programs)}, "
            f"平均长度: {avg_length:.1f}"
        )
 
        return programs, actions, log_probs, entropies, masks
 
 
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
 
    # 5. 查看采样结果
    for i, p in enumerate(programs[:5]):
        print(f"  Program {i}: {p.prefix}  (terminal={p.is_terminal()})")