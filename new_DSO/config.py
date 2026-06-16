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
            'epsilon': 0.05,              # 风险寻求：top-5% 精英
            'use_memory_queue': True,     # 启用历史队列稳定分位数
            'memory_queue_size': 10,      # 历史队列长度
            'memory_decay': 0.9,          # 历史衰减系数
            'baseline_mode': 'R_e',   # 'R_e' | 'ewma_R' | 'ewma_R_e' | 'combined'
        }),
 
        # ---- 数据 ----
        'data': AttrDict({
            'root_type': 'node',       # 'node' 或 'edge'
            'sample_num': 500,         # BFGS 采样点数
            'complexity_base': 0.999,
        }),

        # ---- GP-Meld 遗传规划（第四步新增）----
        'gp': AttrDict({
            'enabled': True,
            'population_size': 50,     # GP 种群大小
            'crossover_rate': 0.7,     # 交叉概率
            'mutation_rate': 0.3,      # 变异概率
            'tournament_size': 5,      # 锦标赛选择大小
            'max_offspring': 50,       # 每轮最大后代数
            'elite_fraction': 0.1,     # 精英保留比例
        }),
    })
    return config