import json
import argparse
import os
import math
from typing import List, Dict


def uniform_split(frames: List[Dict], target_train_size: int = 70) -> tuple:
    """
    均匀间隔划分数据集，固定训练集大小为70张，其余为测试集

    参数：
        frames: 所有帧数据
        target_train_size: 目标训练集大小（默认80）

    返回：
        (train_frames, test_frames)
    """
    n_total = len(frames)
    n_train = min(target_train_size, n_total)  # 确保不超过总帧数
    n_test = n_total - n_train

    # 计算测试帧的均匀间隔
    test_indices = set()
    if n_test > 0:
        interval = n_total / n_test
        for i in range(n_test):
            idx = round(i * interval)
            test_indices.add(min(idx, n_total - 1))

    train_frames = [f for i, f in enumerate(frames) if i not in test_indices]
    test_frames = [f for i, f in enumerate(frames) if i in test_indices]

    return train_frames, test_frames


def main():
    parser = argparse.ArgumentParser(description='均匀间隔划分数据集（固定训练集80张）')
    parser.add_argument('--json_in', default='transforms.json', help='输入transforms.json路径')
    parser.add_argument('--out_dir', default='splits', help='输出目录')
    parser.add_argument('--train_size', type=int, default=70,
                       help='训练集大小（默认80张）')

    args = parser.parse_args()

    # 读取输入文件
    with open(args.json_in, 'r') as f:
        data = json.load(f)

    # 执行均匀划分
    train_frames, test_frames = uniform_split(data['frames'], args.train_size)

    # 保存结果（与原代码相同）
    os.makedirs(args.out_dir, exist_ok=True)
    for split, frames in [('train', train_frames), ('test', test_frames)]:
        output = data.copy()
        output['frames'] = frames
        with open(f'{args.out_dir}/transforms_{split}.json', 'w') as f:
            json.dump(output, f, indent=4)
        print(f'{split}: {len(frames)} frames')

if __name__ == '__main__':
    main()