# new_DSO/run_fish.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import json, numpy as np
from ND2.utils import init_logger, seed_all
from new_DSO.trainer import NewDSOTrainer
from new_DSO.config import get_default_config
 
def main():
    init_logger('new_DSO_FISH', './log/new_dso/fish/info.log', root_name='new_DSO')
    config = get_default_config()
 
    # ---- 基本配置 ----
    config.data.root_type = 'node'
    config.training.entropy_weight = 1.0
    config.training.n_samples = 100000
    config.training.seed = 42
    config.data.complexity_base = 0.999
    config.training.batch_size = 1024
    config.training.epsilon = 0.1
    config.training.baseline_mode = 'ewma_R'
    config.policy.max_length = 30       # 鱼群公式较长，放宽限制
    config.policy.max_coeff_num = 5
 
    # ---- 加载鱼群数据 ----
    data = json.load(open('./data/fish/schooling_nd2.json', 'r'))
    A = np.array(data['A'], dtype=int)
    G = np.array(data['G'], dtype=int)
    x       = np.array(data['x'], dtype=np.float32)
    dx      = np.array(data['dx'], dtype=np.float32)
    M       = np.array(data['M'], dtype=np.float32)
    rho     = np.array(data['rho'], dtype=np.float32)
    phi     = np.array(data['phi'], dtype=np.float32)
    theta_v = np.array(data['theta_vis'], dtype=np.float32)
 
    # ★ 关键：交换 G 的两列，和 search_fish.py 一致
    G = G[:, [1, 0]]
 
    # ---- 定义变量 ----
    Xv = {'v1': x}                           # 节点变量：theta
    Xe = {'e1': M, 'e2': rho, 'e3': phi, 'e4': theta_v}  # 边变量
    Y = dx                                    # 目标：dtheta
 
    # ---- 创建 Trainer 并训练 ----
    trainer = NewDSOTrainer(config=config, Xv=Xv, Xe=Xe, A=A, G=G, Y=Y)
    trainer.reward_solver.sample_num = 200
    trainer.reward_solver.bfgs_max_iter = 15
 
    trainer.fit(
        early_stop_fn=lambda m: m.get('R2', -1) > 0.99,
        checkpoint_path='./log/new_dso/fish/checkpoint.pth',
        log_every=1, save_every=200,
    )
 
if __name__ == '__main__':
    main()