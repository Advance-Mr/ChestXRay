import os
from collections import defaultdict


def count_samples(dataset_path):
    counts = defaultdict(int)
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(split_path, class_name)
            counts[f"{split}_{class_name}"] = len(
                [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # 修正括号闭合
            )
    return dict(counts)


if __name__ == "__main__":
    dataset_path = "chest_xray"
    counts = count_samples(dataset_path)

    print("\n数据集统计结果：")
    print(f"训练集 NORMAL: {counts.get('train_NORMAL', 0)}")
    print(f"训练集 PNEUMONIA: {counts.get('train_PNEUMONIA', 0)}")
    print(f"测试集 NORMAL: {counts.get('test_NORMAL', 0)}")
    print(f"测试集 PNEUMONIA: {counts.get('test_PNEUMONIA', 0)}")
    print("\n请将上述统计结果反馈给我，我将继续优化模型设计")