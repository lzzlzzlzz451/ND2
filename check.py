import json
import numpy as np
from ND2.search.reward_solver import RewardSolver
from ND2.GDExpr import GDExpr
 
# 加载数据
data = json.load(open('data/fish/schooling_nd2.json'))
A = np.array(data['A'])
G = np.array(data['G'])
x = np.array(data['x'], dtype=np.float32)
dx = np.array(data['dx'], dtype=np.float32)
M = np.array(data['M'], dtype=np.float32)
rho = np.array(data['rho'], dtype=np.float32)
phi = np.array(data['phi'], dtype=np.float32)
theta_vis = np.array(data['theta_vis'], dtype=np.float32)
 
# ★ 关键修复：交换 G 的两列
# 原始 G: (src, dst)，边变量从 src 视角计算
# 交换后: (dst, src)，aggr 会按 G[:,1]=src 求和 → 等价于原来的 rgga
G = G[:, [1, 0]]
 
# ★ 用 aggr 替代 rgga
prefix = [
    'div', 'aggr', 'mul', 'M', 'mul', 'add', 'mul', '<C>', 'sin', 'phi',
    'mul', 'rho', 'sin', 'theta_vis', 'add', '1.0', 'cos', 'theta_vis',
    'aggr', 'mul', 'M', 'add', '1.0', 'cos', 'theta_vis'
]
 
# 构建 RewardSolver
rewarder = RewardSolver(
    Xv={'x': x},
    Xe={'M': M, 'rho': rho, 'phi': phi, 'theta_vis': theta_vis},
    A=A, G=G, Y=dx, mask=None,
)
 
# 拟合系数并评估
reward, prefix_with_coef = rewarder.solve(prefix)
metrics = rewarder.evaluate(prefix_with_coef)
 
print(f"拟合表达式: {GDExpr.prefix2str(prefix_with_coef)}")
print(f"Reward: {reward:.6f}")
for k, v in metrics.items():
    print(f"  {k}: {v:.6f}")