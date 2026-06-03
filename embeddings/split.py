import numpy as np
import os
import json
from pathlib import Path
def count_jsonl_samples(jsonl_path):
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count
def split_embeddings(dataset_name, data_dir='../data', embeddings_dir='./'):
    print(f"\n{'='*60}")
    print(f"开始处理数据集: {dataset_name}")
    print(f"{'='*60}")
    data_dir = Path(data_dir)
    embeddings_dir = Path(embeddings_dir)
    train_jsonl = data_dir / f"{dataset_name}_train_data.jsonl"
    test_jsonl = data_dir / f"{dataset_name}_test_data.jsonl"
    if not train_jsonl.exists():
        print(f"错误: 找不到训练文件 {train_jsonl}")
        return False
    if not test_jsonl.exists():
        print(f"错误: 找不到测试文件 {test_jsonl}")
        return False
    train_count = count_jsonl_samples(train_jsonl)
    test_count = count_jsonl_samples(test_jsonl)
    total_count = train_count + test_count
    print(f"训练集样本数: {train_count}")
    print(f"测试集样本数: {test_count}")
    print(f"总样本数: {total_count}")
    source_folder = embeddings_dir / dataset_name
    if not source_folder.exists():
        print(f"错误: 找不到嵌入文件夹 {source_folder}")
        return False
    npy_files = list(source_folder.glob("*.npy"))
    if not npy_files:
        print(f"错误: {source_folder} 中没有找到.npy文件")
        return False
    print(f"\n找到 {len(npy_files)} 个嵌入文件:")
    for npy_file in npy_files:
        print(f"  - {npy_file.name}")
    dataset_folder = embeddings_dir / dataset_name
    train_folder = dataset_folder / f"{dataset_name}-train"
    test_folder = dataset_folder / f"{dataset_name}-test"
    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n创建目标文件夹:")
    print(f"  - {train_folder}")
    print(f"  - {test_folder}")
    print(f"\n开始分割嵌入文件...")
    for npy_file in npy_files:
        print(f"\n处理: {npy_file.name}")
        try:
            embeddings = np.load(npy_file)
        except Exception as e:
            print(f"  错误: 无法加载 {npy_file.name}: {e}")
            print(f"  跳过此文件...")
            continue
        print(f"  原始形状: {embeddings.shape}")
        if embeddings.shape[0] != total_count:
            print(f"  警告: 嵌入样本数 ({embeddings.shape[0]}) 与预期 ({total_count}) 不匹配!")
            print(f"  跳过此文件...")
            continue
        train_embeddings = embeddings[:train_count]
        test_embeddings = embeddings[train_count:]
        print(f"  训练集形状: {train_embeddings.shape}")
        print(f"  测试集形状: {test_embeddings.shape}")
        base_name = npy_file.stem
        if dataset_name in base_name:
            train_name = base_name.replace(dataset_name, f"{dataset_name}_train") + ".npy"
            test_name = base_name.replace(dataset_name, f"{dataset_name}_test") + ".npy"
        else:
            train_name = f"{base_name}_train.npy"
            test_name = f"{base_name}_test.npy"
        train_path = train_folder / train_name
        test_path = test_folder / test_name
        np.save(train_path, train_embeddings)
        np.save(test_path, test_embeddings)
        print(f"  ✓ 保存训练集到: {train_path.name}")
        print(f"  ✓ 保存测试集到: {test_path.name}")
    print(f"\n{'='*60}")
    print(f"数据集 {dataset_name} 处理完成!")
    print(f"{'='*60}")
    return True
def main():
    import argparse
    parser = argparse.ArgumentParser(description='分割嵌入文件为train和test')
    parser.add_argument('--dataset',type=str, required=True,
                        help='数据集名称（如 bbc）')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='数据文件夹路径')
    parser.add_argument('--embeddings_dir', type=str, default='./',
                        help='嵌入文件夹路径')
    args = parser.parse_args()
    print("="*60)
    print("嵌入文件分割工具")
    print("="*60)
    success = split_embeddings(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        embeddings_dir=args.embeddings_dir
    )
    if success:
        print("\n✓ 所有操作成功完成!")
    else:
        print("\n✗ 处理过程中出现错误")
        return 1
    return 0
if __name__ == "__main__":
    exit(main())
