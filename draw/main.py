import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams["axes.unicode_minus"] = False #正常显示负号

# 设置图片大小和像素
plt.figure(figsize=(9, 9), dpi=100)

methods = ['Method1', 'Method2', 'Method3']
# roman-empire
results = [70.53, 66.58, 69.24]
# amazon-ratings
# results = [47.87, 44.06, 46.52] 
# minesweeper
# results = [60.92, 55.97, 57.24]
# tolokers
# results = [78.20, 75.98, 77.21]
# questions
# results = [69.97, 64.67, 66.23]

# 设置柱体样式
labels = ["full", "w/o im1", "w/o im2"]

# 调整柱体的宽度和 x 轴范围
bar_width = 0.7
bar_positions = np.arange(len(methods)) - bar_width / 2

plt.bar([0, 1, 2], results, tick_label=labels, color=['#a3cef1', '#6096ba', '#274c77'], width=bar_width, align='center')

for i, v in enumerate(results):
    plt.text(i, v+0.1, str(round(v,2)), ha='center', fontdict={'fontsize': 30})
# 设置标题
# plt.title("Xxxxxxx", fontsize=16)

# 设置坐标轴名称和字体大小
# plt.xlabel("Method", fontsize=14)
plt.ylabel("Results", fontsize=28)

# 设置坐标轴刻度和字体大小
plt.xticks(range(3), fontsize=32)
plt.yticks(np.arange(0, 90, 16), fontsize=32)

# 设置坐标轴范围
# plt.xlim(bar_positions[0] - 0.5, bar_positions[-1] + 0.5)
plt.ylim(0, 80)

# 设置背景网格:网格线条样式、透明度
# plt.grid(ls='-', alpha=0.8)

# 展示图片
# plt.show()

# 保存图片
plt.savefig('fig/roman-empire.pdf', dpi=600)
