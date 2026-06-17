# new_DSO/run_new_dso.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import json, numpy as np
from ND2.utils import init_logger, seed_all
from new_DSO.trainer import NewDSOTrainer
from new_DSO.config import get_default_config
 
def main():
    init_logger('new_DSO_KUR', './log/new_dso/KUR/info.log', root_name='new_DSO')
    config = get_default_config()
    config.data.root_type = 'node'
    config.training.entropy_weight = 1.0
    config.training.n_samples = 100000
    config.training.seed = 42
    config.data.complexity_base = 0.999
    config.training.batch_size = 1024  # 旧: 256，更多样本
    config.training.epsilon = 0.1    # 旧: 0.05，选 top 10% 作 elite
    config.training.baseline_mode = 'ewma_R'
    config.policy.max_length = 20
    config.gp.n_offspring = 50
    config.gp.n_generations = 3
 
    # 加载 KUR 数据
    data = json.load(open('./data/synthetic/KUR.json', 'r'))
    for k, v in data.items():
        data[k] = np.array(v)
    data['A'] = data['A'].astype(int)
    data['G'] = data['G'].astype(int)
 
    Xv = {'v1': data['x'], 'v2': data['omega']}
    Xe = {}
    Y = data['dx']  # target: dω/dt
 
    trainer = NewDSOTrainer(config=config, Xv=Xv, Xe={}, 
                            A=data['A'], G=data['G'], Y=Y)
    trainer.reward_solver.sample_num = 200
    trainer.reward_solver.bfgs_max_iter = 15
    trainer.fit(
        early_stop_fn=lambda m: m.get('R2', -1) > 0.99,
        checkpoint_path='./log/new_dso/KUR/checkpoint.pth',
        log_every=1, save_every=200,
    )
 
if __name__ == '__main__':
    main()