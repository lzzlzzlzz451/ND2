import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ==========================================
# 1. 配置文件路径和环境参数 (需与生成器一致)
# ==========================================
CSV_FILE = 'ant_swarm_trajectory_data.csv'
NEST_POS = (50.0, 50.0)
FOOD_POS = (85.0, 85.0)
NEST_RADIUS = 5.0
FOOD_RADIUS = 5.0

# 检查数据文件是否存在
if not os.path.exists(CSV_FILE):
    print(f"错误: 找不到数据文件 '{CSV_FILE}'。请先运行刚才的数据生成脚本。")
    exit()

# ==========================================
# 2. 加载并预处理数据
# ==========================================
print(f"正在加载数据 '{CSV_FILE}'...")
df = pd.read_csv(CSV_FILE)
total_frames = df['frame'].max() + 1
num_ants = df['agent_id'].nunique()
print(f"数据加载完成！总帧数: {total_frames}, 蚂蚁数量: {num_ants}")

# ==========================================
# 3. 初始化画布
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect('equal') # 保持XY轴比例一致
ax.set_title("Ant Swarm Trajectory Visualization")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")

# 绘制蚁巢和食物源区域
nest_circle = plt.Circle(NEST_POS, NEST_RADIUS, color='blue', alpha=0.2, label='Nest (Home)')
food_circle = plt.Circle(FOOD_POS, FOOD_RADIUS, color='green', alpha=0.2, label='Food Source')
ax.add_patch(nest_circle)
ax.add_patch(food_circle)

# 标记中心点
ax.plot(*NEST_POS, 'b+', markersize=10)
ax.plot(*FOOD_POS, 'g+', markersize=10)

# 初始化散点图 (数据将在 update 函数中填充)
scatter = ax.scatter([], [], s=15, zorder=5)

# 添加图例 (使用代理图形)
ax.plot([], [], 'o', color='black', label='Foraging (Searching)')
ax.plot([], [], 'o', color='red', label='Carrying Food (Returning)')
ax.legend(loc='upper left')

# ==========================================
# 4. 定义动画更新逻辑
# ==========================================
def init():
    scatter.set_offsets(np.empty((0, 2)))
    return scatter,

def update(frame):
    # 提取当前帧数据
    current_data = df[df['frame'] == frame]
    if current_data.empty:
        return scatter,
    
    # 更新坐标位置
    positions = current_data[['x', 'y']].values
    scatter.set_offsets(positions)
    
    # 更新颜色: 状态 0 (觅食) 为黑色, 状态 1 (归巢) 为红色
    states = current_data['state'].values
    colors = np.where(states == 1, 'red', 'black')
    scatter.set_color(colors)
    
    # 更新标题上的帧数进度
    ax.set_title(f"Ant Swarm Dynamics - Frame {frame:04d} / {total_frames-1:04d}")
    return scatter,

# ==========================================
# 5. 生成并播放动画
# ==========================================
print("正在生成动画，请稍候... (将会弹出一个播放窗口)")
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=total_frames, 
    init_func=init,
    interval=30,     # 每帧间隔 (毫秒)，调小可加快播放速度
    blit=True,       # 仅重绘变化的部分，提升性能
    repeat=True      # 播放结束后循环
)

plt.show()

# 💡 提示：如果你想将动画保存为视频或 GIF，可以取消下面代码的注释
# (需本地预先安装 ffmpeg 或 imagemagick)
# print("正在保存动画为 MP4...")
# ani.save('ant_swarm_simulation.mp4', writer='ffmpeg', fps=30)
# print("保存完成！")