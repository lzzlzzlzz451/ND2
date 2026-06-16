import numpy as np
import scipy.io as sio
import json
import os
from scipy.spatial import Delaunay
from tqdm import tqdm
 
# ===================== 配置 =====================
DATA_DIR = "Ip90_In05_R10_time100_num100_0/"
OUTPUT_FILE = "data/fish/schooling_nd2.json"
DT = 1e-2
N_FISH = 100
# T_TOTAL = 1000
T_TOTAL= 100
SUBSAMPLE = 1
 
# ===================== 工具函数 =====================
 
def get_voronoi_adj(x, y, num):
    location = np.array([x, y])
    tri = Delaunay(location.T)
    v1 = np.ndarray.flatten(tri.simplices)
    v2 = np.ndarray.flatten(tri.simplices[:, [1, 2, 0]])
    vn = np.zeros((num, num))
    vn[v1, v2] = 1
    vn = np.logical_or(vn, vn.T).astype(int)
    np.fill_diagonal(vn, 0)
    return vn
 
def angle_diff(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))
 
# ===================== 第一步：加载全部 .mat 数据 =====================
print("Step 1/4: Loading .mat files...")
all_state = []
all_rdot = []
all_wvision = []
 
for t in tqdm(range(int(T_TOTAL))):
    filepath = DATA_DIR + "{}.mat".format(t)
    data = sio.loadmat(filepath)
    all_state.append(data["state"])
    all_rdot.append(data["rdot"])
    all_wvision.append(data["wvision"])
 
state_list = [all_state[t][:, :, :-1] for t in range(int(T_TOTAL))]
state_list.append(all_state[-1][:, :, -1:])
rdot_list = [all_rdot[t][:, :, :-1] for t in range(int(T_TOTAL))]
rdot_list.append(all_rdot[-1][:, :, -1:])

wvision_list = [all_wvision[t][:, :-1] for t in range(int(T_TOTAL))]
wvision_list.append(all_wvision[-1][:, -1:])
wvision = np.concatenate(wvision_list, axis=1)  # (N_FISH, total_steps)
 
state = np.concatenate(state_list, axis=2)
rdot = np.concatenate(rdot_list, axis=2)
 
x_all     = state[0, :, :]
y_all     = state[1, :, :]
theta_all = state[2, :, :]
vx_all    = rdot[0, :, :]
vy_all    = rdot[1, :, :]
 
# dtheta_all = np.zeros_like(theta_all)
# dtheta_all[:, :-1] = angle_diff(theta_all[:, 1:], theta_all[:, :-1]) / DT

dtheta_all = wvision
 
total_steps = x_all.shape[1]
 
# ===================== 第二步：计算并集邻接矩阵 =====================
print("Step 2/4: Computing adjacency matrices...")
 
sample_indices = list(range(0, total_steps - 1, SUBSAMPLE))
N_samples = len(sample_indices)
 
A_union = np.zeros((N_FISH, N_FISH), dtype=int)
A_per_step = []
 
for idx in tqdm(sample_indices):
    A_t = get_voronoi_adj(x_all[:, idx], y_all[:, idx], N_FISH)
    A_per_step.append(A_t)
    A_union = np.logical_or(A_union, A_t).astype(int)
 
G = np.stack(np.nonzero(A_union), axis=-1)
E = G.shape[0]
V = N_FISH
 
print(f"V={V}, E={E}, N={N_samples}")
 
# ===================== 第三步：提取变量 =====================
print("Step 3/4: Extracting variables...")
 
node_theta  = np.zeros((N_samples, V))
node_dtheta = np.zeros((N_samples, V))
node_pos_x  = np.zeros((N_samples, V))
node_pos_y  = np.zeros((N_samples, V))
node_vx     = np.zeros((N_samples, V))
node_vy     = np.zeros((N_samples, V))
 
edge_rho       = np.zeros((N_samples, E))
edge_phi       = np.zeros((N_samples, E))
edge_theta_vis = np.zeros((N_samples, E))
M              = np.zeros((N_samples, E))
 
src = G[:, 0]
dst = G[:, 1]
 
for n, idx in enumerate(tqdm(sample_indices)):
    xi  = x_all[:, idx]
    yi  = y_all[:, idx]
    thi = theta_all[:, idx]
    vxi = vx_all[:, idx]
    vyi = vy_all[:, idx]
    dthi = dtheta_all[:, idx]
 
    node_theta[n]  = thi
    node_dtheta[n] = dthi
    node_pos_x[n]  = xi
    node_pos_y[n]  = yi
    node_vx[n]     = vxi
    node_vy[n]     = vyi
 
    dx_e = x_all[dst, idx] - x_all[src, idx]
    dy_e = y_all[dst, idx] - y_all[src, idx]
    edge_rho[n]       = np.sqrt(dx_e**2 + dy_e**2)
    edge_phi[n]       = angle_diff(thi[dst], thi[src])
    bearing           = np.arctan2(dy_e, dx_e)
    edge_theta_vis[n] = angle_diff(bearing, thi[src])
 
    A_t = A_per_step[n]
    M[n] = A_t[src, dst].astype(float)
 
# ===================== 第四步：直接组装 JSON =====================
print("Step 4/4: Assembling JSON (no normalization)...")
 
# 安全检查
for name, arr in [
    ("theta", node_theta), ("dtheta", node_dtheta),
    ("pos_x", node_pos_x), ("pos_y", node_pos_y),
    ("vx", node_vx), ("vy", node_vy),
    ("rho", edge_rho), ("phi", edge_phi),
    ("theta_vis", edge_theta_vis), ("M", M),
    ("A", A_union), ("G", G),
]:
    assert not np.any(np.isnan(arr)), f"{name} contains NaN!"
    assert not np.any(np.isinf(arr)), f"{name} contains Inf!"
 
data_dict = {
    "A": A_union.tolist(),
    "G": G.tolist(),
    "x": node_theta.tolist(),
    "dx": node_dtheta.tolist(),
    "M": M.tolist(),
    "pos_x": node_pos_x.tolist(),
    "pos_y": node_pos_y.tolist(),
    "vx": node_vx.tolist(),
    "vy": node_vy.tolist(),
    "rho": edge_rho.tolist(),
    "phi": edge_phi.tolist(),
    "theta_vis": edge_theta_vis.tolist(),
}
 
with open(OUTPUT_FILE, "w") as f:
    json.dump(data_dict, f)
 
print(f"\nSaved to {OUTPUT_FILE}")
print(f"Dimensions: V={V}, E={E}, N={N_samples}")
print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")