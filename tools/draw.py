import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 设置全局字体大小
plt.rcParams.update({'font.size': 14})  # 增加字体大小

# 加载特征和标签
features = np.load("/data1/fast-reid/tools/output/extracted_features.npy")
targets = np.load("/data1/fast-reid/tools/output/extracted_targets.npy")

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, perplexity=5, n_iter=3000, random_state=42)
features_tsne = tsne.fit_transform(features)

# 绘制聚类结果，使用标签着色，并调整点的透明度
plt.figure(figsize=(10, 6))

# 获取所有唯一的行人ID及其颜色映射
unique_labels = np.unique(targets)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

# 为每个行人ID绘制散点图并添加图例条目
for label, color in zip(unique_labels, colors):
    indices = targets == label
    plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], color=color, label=f"ID: {label}", alpha=0.5)

plt.title('2D Clustering Visualization with t-SNE')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

# 调整图例位置和大小
plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='medium', title="Person ID", title_fontsize='medium')
plt.tight_layout()

# 保存图片
plt.savefig('/data1/fast-reid/tools/output/cluster_visualization.png', dpi=600)
plt.show()
