import numpy as np
from scipy.spatial import Delaunay
 
# ===== 从你的模拟代码复制这些函数 =====
def getangle(phi, rhox, rhoy, rhom=None):
    if rhom is None:
        rhom = np.sqrt(rhox**2 + rhoy**2)
    rhox = rhox / rhom
    rhoy = rhoy / rhom
    ex = np.cos(phi)
    ey = np.sin(phi)
    sgn = np.array(np.sign(ex * rhoy - ey * rhox))
    sgn[sgn == 0] = 1
    return sgn * np.arccos(np.clip(ex * rhox + ey * rhoy, -1, 1))
 
def GetVoronoi(location, num):
    tri = Delaunay(np.transpose(location))
    v1 = np.ndarray.flatten(tri.simplices)
    v2 = np.ndarray.flatten(tri.simplices[:, [1, 2, 0]])
    vn = np.zeros((num, num))
    vn[v1, v2] = 1
    vn = np.logical_or(vn, vn.T)
    return vn
 
# ===== 加载模拟数据 =====
import scipy.io
mat = scipy.io.loadmat('你的模拟输出.mat')
state = mat['state']  # shape: (3, num, steps)
 
num = state.shape[1]
steps = state.shape[2]
Ip = 9.0  # 你的参数
 
# ===== 逐步验证 =====
errors = []
for t in range(steps):
    x = state[0, :, t]
    y = state[1, :, t]
    a = state[2, :, t]
    
    # 模拟中的实际 dx（角速度）
    if t < steps - 1:
        dx_actual = (state[2, :, t+1] - state[2, :, t]) / 1e-2  # 除以dt
    else:
        continue
    
    # 用公式重新计算 wVision
    location = np.array([x, y])
    vn = GetVoronoi(location, num)
    
    listI = np.tile(np.arange(0, num), (num,))
    ns = listI[np.ndarray.flatten(vn)]
    nn = np.sum(vn, axis=0)
    nnmax = np.amax(nn)
    
    neighborI = np.zeros((num, nnmax + 1))
    neighborI[np.arange(0, num), nn] = num
    neighborI = np.cumsum(neighborI[:, :-1], axis=1).astype(int)
    neighborI[neighborI == 0] = ns
    
    xN = np.append(x, np.nan)
    yN = np.append(y, np.nan)
    aN = np.append(a, np.nan)
    
    phi = aN[neighborI] - a[:, None]
    rho1 = xN[neighborI] - x[:, None]
    rho2 = yN[neighborI] - y[:, None]
    rhon = np.sqrt(rho1**2 + rho2**2)
    theta = getangle(a[:, None], rho1, rho2, rhon)
    
    visual = 1 + np.cos(theta)
    dx_formula = np.nansum(
        (Ip * np.sin(phi) + rhon * np.sin(theta)) * visual, axis=1
    ) / np.nansum(visual, axis=1)
    
    # 比较
    residual = dx_actual - dx_formula
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((dx_actual - np.mean(dx_actual))**2)
    r2 = 1 - ss_res / ss_tot
    errors.append(r2)
 
print(f"Mean R2 across time steps: {np.mean(errors):.6f}")
print(f"Min R2: {np.min(errors):.6f}")
print(f"Max R2: {np.max(errors):.6f}")