import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(confusion_matrix, class_names, figsize=(12, 10)):

    plt.figure(figsize=figsize, dpi=400)

    ax = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        cbar=True,
        annot_kws={'size': 10, 'color': 'black'},
        linewidths=0,
        linecolor='grey'
    )

    # 设置坐标轴标签
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')

    # 设置刻度标签
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, rotation=0, fontsize=10)

    # 添加标题
    plt.title('Confusion Matrix', fontsize=14, pad=20, fontweight='bold')

    # 优化布局
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()



# 示例用法
if __name__ == "__main__":

    confusion_matrix = np.array([
        [1033, 201, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [313, 5537, 206, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 106, 765, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 20, 9, 3969, 103, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 140, 152, 152, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 101, 78, 3899, 150, 43, 0, 62, 0, 0, 0, 0, 0],
        [0, 0, 0, 7, 4, 490, 155, 172, 3, 68, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 1, 163, 156, 176, 43, 613, 41, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 31, 60, 137, 51, 748, 76, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 96, 16, 30, 35, 3874, 687, 44, 54, 0, 0],
        [0, 0, 0, 0, 0, 11, 4, 40, 30, 1014, 5378, 509, 76, 0, 4],
        [0, 0, 0, 0, 0, 3, 2, 5, 0, 42, 322, 1519, 241, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 5, 0, 6, 91, 316, 1761, 149, 87],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 217, 509, 716],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 152, 187, 2065]
    ])
    normalized_cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    num_classes = 15

    class_names = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8","t9+", "tM", "tSB", "tB", "tEB"]

    plot_confusion_matrix(normalized_cm, class_names)