import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class tsne_Visualization():
    def __init__(self, features, labels, title=None):
        self.features = features
        self.labels = labels
        self.title = title

    def plot_tsne(self, save=False, fig_name=None):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        # 使用 TSNE 对特征降维
        tsne = TSNE(n_components=2, init='pca', random_state=42)
        features = tsne.fit_transform(self.features)

        # 设置画布大小和分辨率
        plt.figure(figsize=(6, 6), dpi=300)

        # 绘制散点图
        for i in range(features.shape[0]):
            plt.scatter(features[i, 0], features[i, 1], s=7, color=plt.cm.Paired(self.labels[i]))

        # 设置标题（如果有的话）
        if self.title is not None:
            plt.title(self.title)

        # 取消坐标轴刻度
        plt.gca().tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # 获取当前图形的边界范围
        x_min, x_max = features[:, 0].min(), features[:, 0].max()
        y_min, y_max = features[:, 1].min(), features[:, 1].max()
        plot_range = max(x_max - x_min, y_max - y_min)  # 获取最大范围
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # 设置为正方形范围
        plt.xlim([x_center - plot_range / 2, x_center + plot_range / 2])
        plt.ylim([y_center - plot_range / 2, y_center + plot_range / 2])

        # 设置轴的比例
        plt.gca().set_aspect('equal', adjustable='box')

        # 紧凑布局
        plt.tight_layout()

        # 保存或显示图像
        if save and fig_name is not None:
            plt.savefig(fig_name, bbox_inches='tight')
        plt.show()

