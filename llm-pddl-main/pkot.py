import matplotlib.pyplot as plt
import numpy as np

# 模拟数据：10个任务，每个任务包含耗时（分钟）和是否成功（0=失败，1=成功）
np.random.seed(42)
tasks = {
    "Task": [f"Task {i+1}" for i in range(10)],
    "Time": np.random.randint(5, 120, size=10),  # 耗时范围5~120分钟
    "Success": np.random.choice([0, 1], size=10)  # 随机生成成功/失败
}

# 创建散点图
plt.figure(figsize=(10, 6))
colors = ['red' if s == 0 else 'green' for s in tasks["Success"]]
scatter = plt.scatter(tasks["Task"], tasks["Time"], c=colors, s=100, alpha=0.7)

# 添加标注和样式
plt.title("任务规划结果分析", fontsize=14, pad=20)
plt.xlabel("任务名称", fontsize=12)
plt.ylabel("耗时（分钟）", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 添加图例
legend_labels = {0: "失败", 1: "成功"}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10)
           for color in ['red', 'green']]
plt.legend(handles, legend_labels.values(), title="状态")

plt.tight_layout()
plt.show()
