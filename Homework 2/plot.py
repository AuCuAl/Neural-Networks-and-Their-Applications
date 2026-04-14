import matplotlib.pyplot as plt
import json
import os
import glob

# --- 目录配置 ---
HISTORY_DIR = './history' # 确保你在 train_and_eval.py 中生成的 json 都放在了这个目录下
FIGURES_DIR = './Figures'
AUGMENTED_HISTORY_DIR = f'{HISTORY_DIR}/augmented'

# 自动创建 Figures 目录（如果不存在）
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_all_models():
    # 自动获取 history 目录下所有的 json 文件
    json_files = glob.glob(os.path.join(HISTORY_DIR, '*.json'))

    if not json_files:
        print(f"⚠️ 在 {HISTORY_DIR} 目录下没有找到任何 .json 文件，请检查路径！")
        return

    # 创建画布 (稍微加宽一点，给图例和标注留出空间)
    plt.figure(figsize=(15, 6))

    # 获取 Matplotlib 预设的高对比度颜色列表 (tab10 包含10种极具区分度的颜色)
    colors = plt.cm.tab10.colors

    for idx, file_path in enumerate(json_files):
        # 从文件名中提取模型名称，自动兼容带有 "_history" 后缀的情况
        base_name = os.path.basename(file_path)
        model_name = base_name.replace('_history.json', '').replace('.json', '')

        # 为当前模型分配一种固定颜色
        color = colors[idx % len(colors)]

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        epochs = range(1, len(history['train_loss']) + 1)
        last_epoch = epochs[-1]

        # ==========================================
        # 1. 绘制损失曲线 (左侧)
        # ==========================================
        plt.subplot(1, 2, 1)
        # 实线: 训练集 | 虚线: 测试集
        plt.plot(epochs, history['train_loss'], linestyle='-', color=color, label=f'{model_name} (Train)')
        plt.plot(epochs, history['test_loss'], linestyle='--', color=color, label=f'{model_name} (Test)')

        # 标注最后一次 epoch 的 Test Loss
        last_test_loss = history['test_loss'][-1]
        plt.annotate(f"{last_test_loss:.3f}",
                     xy=(last_epoch, last_test_loss),
                     xytext=(5, 0), textcoords='offset points',
                     color=color, fontsize=9, va='center')

        # ==========================================
        # 2. 绘制准确率曲线 (右侧)
        # ==========================================
        plt.subplot(1, 2, 2)
        # 实线: 训练集 | 虚线: 测试集
        plt.plot(epochs, history['train_acc'], linestyle='-', color=color, label=f'{model_name} (Train)')
        plt.plot(epochs, history['test_acc'], linestyle='--', color=color, label=f'{model_name} (Test)')

        # 标注最后一次 epoch 的 Test Acc
        last_test_acc = history['test_acc'][-1]
        plt.annotate(f"{last_test_acc:.1f}%",
                     xy=(last_epoch, last_test_acc),
                     xytext=(5, 0), textcoords='offset points',
                     color=color, fontsize=9, va='center')

        # (智能优化) 找出测试集准确率的最高点并用星号标注
        max_test_acc = max(history['test_acc'])
        max_acc_epoch = epochs[history['test_acc'].index(max_test_acc)]

        # 如果最高点不是最后一个点，我们才额外标注它，防止重叠
        if max_acc_epoch != last_epoch:
            plt.plot(max_acc_epoch, max_test_acc, marker='*', color=color, markersize=8)
            plt.annotate(f"Max: {max_test_acc:.1f}%",
                         xy=(max_acc_epoch, max_test_acc),
                         xytext=(-20, 10), textcoords='offset points',
                         color=color, fontsize=9)

    # --- 完善左图细节 ---
    plt.subplot(1, 2, 1)
    plt.title('Train and Test Loss Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7) # 加入虚线网格线更易读数据

    # --- 完善右图细节 ---
    plt.subplot(1, 2, 2)
    plt.title('Train and Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()

    # 保存高斯清晰度的图表
    save_path = os.path.join(FIGURES_DIR, 'all_models_comparison.png')
    # dpi=300 保证插入到 Word 报告或 PDF 中时放大不会模糊
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 所有模型的对比曲线图已成功生成并保存至: {save_path}")

    # 弹出窗口显示
    plt.show()

def plot_augmentation_comparison():
    # 两个实验的历史文件路径
    base_file = os.path.join(HISTORY_DIR, 'SimpleCNN.json')
    aug_file = os.path.join(AUGMENTED_HISTORY_DIR, 'SimpleCNN_augmented.json')

    # 检查文件是否存在
    if not os.path.exists(base_file) or not os.path.exists(aug_file):
        print(f"⚠️ 找不到 JSON 文件，请检查 {HISTORY_DIR} 目录下是否包含这两个文件！")
        return

    # 读取数据
    with open(base_file, 'r', encoding='utf-8') as f:
        history_base = json.load(f)
    with open(aug_file, 'r', encoding='utf-8') as f:
        history_aug = json.load(f)

    epochs = range(1, len(history_base['train_loss']) + 1)
    last_epoch = epochs[-1]

    # --- 颜色与线型设定 ---
    # 蓝色代表 Baseline(原版)，橙色代表 Augmented(数据增强版)
    color_base = '#1f77b4'  # 标准蓝
    color_aug = '#ff7f0e'   # 标准橙

    plt.figure(figsize=(14, 6))

    # ==========================================
    # 1. 绘制损失曲线 (左侧)
    # ==========================================
    plt.subplot(1, 2, 1)

    # Baseline 曲线
    plt.plot(epochs, history_base['train_loss'], linestyle='-', color=color_base, label='Baseline (Train)')
    plt.plot(epochs, history_base['test_loss'], linestyle='--', color=color_base, label='Baseline (Test)')

    # Augmented 曲线
    plt.plot(epochs, history_aug['train_loss'], linestyle='-', color=color_aug, label='Augmented (Train)')
    plt.plot(epochs, history_aug['test_loss'], linestyle='--', color=color_aug, label='Augmented (Test)')

    # 标注最后一次 Epoch 的测试集 Loss
    base_last_test_loss = history_base['test_loss'][-1]
    aug_last_test_loss = history_aug['test_loss'][-1]

    plt.annotate(f"{base_last_test_loss:.3f}", xy=(last_epoch, base_last_test_loss),
                 xytext=(5, 0), textcoords='offset points', color=color_base, fontsize=9, va='center')
    plt.annotate(f"{aug_last_test_loss:.3f}", xy=(last_epoch, aug_last_test_loss),
                 xytext=(5, 0), textcoords='offset points', color=color_aug, fontsize=9, va='center')

    plt.title('Loss Comparison: Baseline vs Data Augmentation', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.7)

    # ==========================================
    # 2. 绘制准确率曲线 (右侧)
    # ==========================================
    plt.subplot(1, 2, 2)

    # Baseline 曲线
    plt.plot(epochs, history_base['train_acc'], linestyle='-', color=color_base, label='Baseline (Train)')
    plt.plot(epochs, history_base['test_acc'], linestyle='--', color=color_base, label='Baseline (Test)')

    # Augmented 曲线
    plt.plot(epochs, history_aug['train_acc'], linestyle='-', color=color_aug, label='Augmented (Train)')
    plt.plot(epochs, history_aug['test_acc'], linestyle='--', color=color_aug, label='Augmented (Test)')

    # 标注最后一次 Epoch 的测试集 Acc
    base_last_test_acc = history_base['test_acc'][-1]
    aug_last_test_acc = history_aug['test_acc'][-1]

    plt.annotate(f"{base_last_test_acc:.1f}%", xy=(last_epoch, base_last_test_acc),
                 xytext=(5, 0), textcoords='offset points', color=color_base, fontsize=9, va='center')
    plt.annotate(f"{aug_last_test_acc:.1f}%", xy=(last_epoch, aug_last_test_acc),
                 xytext=(5, 0), textcoords='offset points', color=color_aug, fontsize=9, va='center')

    # (智能优化) 寻找并标注各自的最高准确率
    for hist, col, name in zip([history_base, history_aug], [color_base, color_aug], ['Base', 'Aug']):
        max_acc = max(hist['test_acc'])
        max_epoch = epochs[hist['test_acc'].index(max_acc)]
        if max_epoch != last_epoch:
            plt.plot(max_epoch, max_acc, marker='*', color=col, markersize=8)
            # 错开一点位置防止文字重叠
            offset = 10 if name == 'Aug' else -15
            plt.annotate(f"Max: {max_acc:.1f}%", xy=(max_epoch, max_acc),
                         xytext=(-20, offset), textcoords='offset points', color=col, fontsize=9)

    plt.title('Accuracy Comparison: Baseline vs Data Augmentation', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()

    # 保存高清对比图
    save_path = os.path.join(FIGURES_DIR, 'augmentation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 数据增强对比图已生成并保存至: {save_path}")

    plt.show()

if __name__ == '__main__':
    plot_all_models()
    plot_augmentation_comparison()