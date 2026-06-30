import os
import json
import time
import signal
import logging
import warnings
import traceback
import numpy as np
from argparse import ArgumentParser
from setproctitle import setproctitle
from ND2.model import NDformer
from ND2.utils import init_logger, AutoGPU, seed_all
from ND2.search import MCTS
from ND2.GDExpr import GDExpr
from ND2.search.reward_solver import RewardSolver
 
warnings.filterwarnings("ignore", category=RuntimeWarning)
def handler(signum, frame): raise KeyboardInterrupt    
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
logger = logging.getLogger('ND2.search')
 
 
class FishRewardSolver(RewardSolver):
    def solve(self, prefix, *args, **kwargs):
        n_aggr = prefix.count('aggr')
        if n_aggr < 1 or n_aggr > 2: return 0.0, {}  # ← 允许1或2个aggr
        reward, coef_dict = super().solve(prefix, *args, **kwargs)
        return reward, coef_dict
 
 
def main(args):
    init_logger(args.name, f'./log/fish/{args.name}/info.log', root_name='ND2', info_level=args.info_level)
    setproctitle(f'{args.name}@FishSchooling')
    if args.seed is None: args.seed = np.random.randint(0, 32768)
    seed_all(args.seed)
    if args.device == 'auto': args.device = AutoGPU().choice_gpu(900, interval=15, force=True)
    logger.info(f'Args: {args}')
 
    # %% 第一步：加载数据
    data = json.load(open(args.data, 'r'))
    A = np.array(data['A'], dtype=int)
    G = np.array(data['G'], dtype=int)
    x       = np.array(data['x'], dtype=np.float32)         # theta
    dx      = np.array(data['dx'], dtype=np.float32)        # dtheta（干净的 wVision）
    M       = np.array(data['M'], dtype=np.float32)         # Voronoi 邻接掩码
    rho     = np.array(data['rho'], dtype=np.float32)       # 距离
    phi     = np.array(data['phi'], dtype=np.float32)       # 相对朝向角
    theta_v = np.array(data['theta_vis'], dtype=np.float32) # 视角
 
    # ★ 关键：交换 G 的两列，使 aggr 按源节点求和（等价于 rgga）
    G = G[:, [1, 0]]

    Xv = {'v1': x}                                          # 节点变量：v1 = theta
    Xe = {'e1': M, 'e2': rho, 'e3': phi, 'e4': theta_v} 
    # %% 第二步：初始化 RewardSolver
    rewarder = FishRewardSolver(
        Xv=Xv,                                          # 节点变量
        Xe=Xe,  # 边变量
        A=A, G=G, Y=dx,
        mask=None,
    )
 
    # %% 第三步：初始化 NDformer（预训练策略网络）
    ndformer = NDformer(device=args.device)
    ndformer.load(args.model_path, weights_only=False)
    ndformer.eval()
    ndformer.set_data(
        Xv=Xv,
        Xe=Xe,
        A=A, G=G, Y=dx,
        root_type='node',
        cache_data_emb=True,
    )
 
    # %% 第四步：初始化 MCTS
    est = MCTS(
        rewarder=rewarder,
        ndformer=ndformer,
        vars_node=['v1'],                                      # 节点变量名
        vars_edge=['e1', 'e2', 'e3', 'e4'],           # 边变量名
        binary=['add', 'sub', 'mul', 'div', 'regular'],       # 二元算子
        unary=['neg', 'abs', 'inv', 'exp', 'logabs', 'sqrtabs',
               'pow2', 'pow3', 'sin', 'cos', 'tanh', 'sigmoid',
               'aggr', 'sour', 'targ'],                       # 一元算子（含图算子）
        constant=['1'],                                     # 常量
        log_per_episode=None,
        log_per_second=10,
        beam_size=10,
        use_random_simulate=False,
        max_token_num=30,                                     # 表达式最大长度
        max_coeff_num=5,                                      # 最大系数个数
    )
 
    # %% 可选：验证 ground truth 作为 baseline
    if args.eval_baseline:
        gt_prefix = [
            'div', 'aggr', 'mul', 'e1', 'mul', 'add', 'mul', '<C>', 'sin', 'e3',
            'mul', 'e2', 'sin', 'e4', 'add', '1', 'cos', 'e4',
            'aggr', 'mul', 'e1', 'add', '1', 'cos', 'e4'
        ]
        # 直接传已知系数 Ip=9.0，不需要先 solve
        metrics = rewarder.evaluate(gt_prefix, {'<C>': [9.0]})
        log = {'Baseline-GT': GDExpr.prefix2str(gt_prefix), **metrics}
        logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))
 
    # %% 第五步：搜索
    try:
        logger.note('Start searching... Press ^C (Ctrl+C) to stop when satisfied.')
        est.fit(
            ['node'],
            episode_limit=100_000_000,
            time_limit=args.time_limit,
            early_stop=None,
        )
    except KeyboardInterrupt:
        logger.info('Interrupted manually.')
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        log = {
            'Discovered': GDExpr.prefix2str(est.best_model),
            **est.best_metric,
        }
        logger.note(' | '.join(f'\033[4m{k}\033[0m:{v}' for k, v in log.items()))
        pareto = est.Pareto()
 
        # 保存结果
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        with open(args.save_path, 'a') as f:
            json.dump(dict(
                name=args.name,
                seed=args.seed,
                result=est.best_model,
                **est.best_metric,
            ), f)
            f.write('\n')
 
 
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=f'FishSearch_{time.strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('-d', '--device', type=str, default='auto')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('--data', type=str, default='./data/fish/schooling_nd2.json')
    parser.add_argument('--model_path', type=str, default='./weights/checkpoint.pth')
    parser.add_argument('--time_limit', type=int, default=86400)
    parser.add_argument('--save_path', type=str, default='./result/fish_search.json')
    parser.add_argument('--eval_baseline', action='store_true', help='评估 ground truth 公式作为 baseline')
    parser.add_argument('--info_level', choices=['debug', 'info', 'note', 'warning', 'error', 'critical'], default='note')
    args, unknown = parser.parse_known_args()
    if unknown: warnings.warn(f'Unknown args: {unknown}')
    main(args)