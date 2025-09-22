#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json

# 从文件中读取数据
data_points = []
with open('/home/shane/code_b/unified_video_action/act_diff_testing_steps.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            # 解析每行数据
            # 格式: "act_diff_testing_steps": 100, "test/mean_score": 0.9544941507868598;
            parts = line.split(',')
            steps_part = parts[0].split(':')[1].strip()
            score_part = parts[1].split(':')[1].strip().rstrip(';')
            
            steps = int(steps_part)
            score = float(score_part)
            data_points.append((steps, score))

# 按steps排序
data_points.sort(key=lambda x: x[0])

# 提取x和y坐标
x_values = [point[0] for point in data_points]
y_values = [point[1] for point in data_points]

# 创建图形
plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values, 'bo-', linewidth=2, markersize=8, markerfacecolor='red', markeredgecolor='darkred')

# 设置图形属性
plt.xlabel('act_diff_testing_steps', fontsize=14, fontweight='bold')
plt.ylabel('test/mean_score', fontsize=14, fontweight='bold')
plt.title('Test Mean Score vs Act Diff Testing Steps', fontsize=16, fontweight='bold')

# 设置坐标轴范围
plt.xlim(0, 105)
plt.ylim(min(y_values) - 0.05, max(y_values) + 0.05)

# 添加网格
plt.grid(True, alpha=0.3)

# 设置x轴刻度
plt.xticks(range(0, 101, 10))

# 添加数据点标签（可选，如果数据点不多的话）
for i, (x, y) in enumerate(data_points):
    plt.annotate(f'({x}, {y:.3f})', (x, y), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=8)

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig('/home/shane/code_b/unified_video_action/curve_plot.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()

print("曲线图已保存为 curve_plot.png")
print(f"数据点数量: {len(data_points)}")
print("数据点:")
for x, y in data_points:
    print(f"  act_diff_testing_steps: {x}, test/mean_score: {y:.6f}")
