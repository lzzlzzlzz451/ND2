# new_DSO/run_fish_dso.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import json, numpy as np, torch
from ND2.utils import init_logger, seed_all
from new_DSO.trainer import NewDSOTrainer
from new_DSO.config import get_default_config
 
def main():
    init_logger('new_DSO_Fish', './log/new_dso/fish/info.log', root_name='new_DSO')
 
    config = get_default_config()
    config.data.root_type = 'node'
    config.training.seed = 42
    config.training.batch_size = 256
    config.training.n_samples = 100000
    config.training.entropy_weight = 0.5
    config.training.epsilon = 0.05 
    config.training.baseline_mode = 'ewma_R'
    config.training.learning_rate = 1e-4
    config.data.complexity_base = 0.999
    config.data.sample_num = 500
 
    config.policy.max_length = 30
    config.policy.d_model = 128
    config.policy.nhead = 4
    config.policy.num_layers = 3
    config.policy.dim_feedforward = 256
 
    # 编码器配置与 NDformer 对齐
    config.encoder.n_GNN_layers = 2
    config.encoder.n_transformer_layers = 2
    config.encoder.n_node_vars = 6
    config.encoder.n_edge_vars = 6
    config.encoder.d_data_feat = 16
    config.encoder.max_sample_num = 3000
    config.encoder.freeze = True
 
    # GP 配置
    config.gp.enabled = True
    config.gp.population_size = 256
    config.gp.crossover_rate = 0.7
    config.gp.mutation_rate = 0.3
    config.gp.max_offspring = 256
 
    # ---- 加载鱼群数据 ----
    data_path = './data/fish/schooling_nd2.json'
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        print("请先运行: python gen_fish_data.py")
        return
 
    data = json.load(open(data_path, 'r'))
    A = np.array(data['A'], dtype=int)
    G = np.array(data['G'], dtype=int)
    # ★ 交换 G 两列，与 search_fish.py 保持一致
    G = G[:, [1, 0]]
 
    x       = np.array(data['x'], dtype=np.float32)          # theta
    dx      = np.array(data['dx'], dtype=np.float32)         # dtheta
    M       = np.array(data['M'], dtype=np.float32)          # Voronoi mask
    rho     = np.array(data['rho'], dtype=np.float32)        # 距离
    phi     = np.array(data['phi'], dtype=np.float32)        # 相对朝向角
    theta_v = np.array(data['theta_vis'], dtype=np.float32)  # 视角
 
    Xv = {'v1': x}
    Xe = {'e1': M, 'e2': rho, 'e3': phi, 'e4': theta_v}
    Y = dx
 
    V, E = A.shape[0], G.shape[0]
    N = Y.shape[0]
    print(f"鱼群数据: V={V}, E={E}, N={N}")
    print(f"  节点变量: {list(Xv.keys())}")
    print(f"  边变量: {list(Xe.keys())}")
    print(f"  Y range: [{Y.min():.4f}, {Y.max():.4f}]")
 
    # ---- 训练器 ----
    trainer = NewDSOTrainer(
        config=config,
        Xv=Xv, Xe=Xe,
        A=A, G=G, Y=Y,
    )
 
    # 调整奖励计算器（鱼群数据量较大，适当降低采样）
    trainer.reward_solver.sample_num = 200
    trainer.reward_solver.bfgs_max_iter = 15
 
    # ---- 开始搜索 ----
    trainer.fit(
        early_stop_fn=lambda m: m.get('R2', -1) > 0.95,
        checkpoint_path='./log/new_dso/fish/checkpoint.pth',
        log_every=1,
        save_every=200,
    )
 
if __name__ == '__main__':
    main()