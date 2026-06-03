import numpy as np
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
def calculate_avg_length(texts):
    total_length = sum(len(str(text)) for text in texts)
    return total_length / len(texts) if len(texts) > 0 else 0
def analyze_dataset(file_path):
    print(f"\n正在分析: {file_path}")
    try:
        if file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=True)
            texts = data['data']
            labels = data['label']
        elif file_path.endswith('.npy'):
            loaded_data = np.load(file_path, allow_pickle=True)
            if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
                if len(loaded_data) > 0 and isinstance(loaded_data[0], dict):
                    texts = [item.get('text', item.get('data', '')) for item in loaded_data]
                    labels = [item.get('label', 0) for item in loaded_data]
                else:
                    texts = loaded_data
                    labels = np.zeros(len(texts))
            else:
                texts = loaded_data
                labels = np.zeros(len(texts))
        else:
            print(f"  跳过: 不支持的文件格式")
            return None
        total_samples = len(texts)
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))
        normal_count = label_dist.get(0, 0)
        anomaly_count = label_dist.get(1, 0)
        if set(unique_labels) - {0, 1}:
            print(f"  注意: 检测到非0/1标签，标签分布: {label_dist}")
            min_label = min(unique_labels)
            normal_count = label_dist.get(min_label, 0)
            anomaly_count = total_samples - normal_count
        avg_length = calculate_avg_length(texts)
        stats = {
            'file_name': os.path.basename(file_path),
            'total_samples': total_samples,
            'normal_samples': normal_count,
            'anomaly_samples': anomaly_count,
            'avg_sentence_length': avg_length,
            'total_chars': total_samples * avg_length,
            'anomaly_ratio': (anomaly_count / total_samples * 100) if total_samples > 0 else 0,
            'label_distribution': label_dist
        }
        return stats
    except Exception as e:
        import traceback
        print(f"  错误: {str(e)}")
        traceback.print_exc()
        return None

def print_stats(stats):
    print(f"\n{'='*70}")
    print(f"数据集: {stats['file_name']}")
    print(f"{'='*70}")
    print(f"总样本数:        {stats['total_samples']:,}")
    print(f"正常样本数:      {stats['normal_samples']:,} ({stats['normal_samples']/stats['total_samples']*100:.2f}%)")
    print(f"异常样本数:      {stats['anomaly_samples']:,} ({stats['anomaly_ratio']:.2f}%)")
    print(f"平均句子长度:    {stats['avg_sentence_length']:.2f} 字符")
    print(f"总字符数:        {stats['total_chars']:,.0f} (样本数×平均长度)")
    print(f"标签分布:        {stats['label_distribution']}")

def save_to_excel(all_stats, output_path):
    if not all_stats:
        print("没有数据可以保存")
        return
    df_data = []
    for stats in all_stats:
        df_data.append({
            '数据集名称': stats['file_name'],
            '总样本数': stats['total_samples'],
            '正常样本数': stats['normal_samples'],
            '异常样本数': stats['anomaly_samples'],
            '正常样本占比(%)': f"{stats['normal_samples']/stats['total_samples']*100:.2f}",
            '异常样本占比(%)': f"{stats['anomaly_ratio']:.2f}",
            '平均句子长度(字符)': f"{stats['avg_sentence_length']:.2f}",
            '总字符数': int(stats['total_chars']),
            '标签分布': str(stats['label_distribution'])
        })
    df = pd.DataFrame(df_data)
    total_samples = df['总样本数'].sum()
    total_normal = df['正常样本数'].sum()
    total_anomaly = df['异常样本数'].sum()
    total_chars = df['总字符数'].sum()
    summary_row = pd.DataFrame([{
        '数据集名称': '总计',
        '总样本数': total_samples,
        '正常样本数': total_normal,
        '异常样本数': total_anomaly,
        '正常样本占比(%)': f"{total_normal/total_samples*100:.2f}",
        '异常样本占比(%)': f"{total_anomaly/total_samples*100:.2f}",
        '平均句子长度(字符)': '-',
        '总字符数': total_chars,
        '标签分布': '-'
    }])
    df = pd.concat([df, summary_row], ignore_index=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='数据集统计', index=False)
        worksheet = writer.sheets['数据集统计']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(col)
            ) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)
    print(f"\n✓ 统计结果已保存到: {output_path}")

def main():
    print("="*70)
    print("NLP异常检测数据集统计分析")
    print("="*70)
    data_dir = Path(__file__).parent
    npz_files = list(data_dir.glob("*.npz"))
    npy_files = list(data_dir.glob("*.npy"))
    all_files = npz_files + npy_files
    if not all_files:
        print("\n未找到任何.npz或.npy数据集文件")
        print("请确保数据集文件已上传到data目录")
        return
    print(f"\n找到 {len(all_files)} 个数据集文件:")
    for f in all_files:
        print(f"  - {f.name}")
    all_stats = []
    for file_path in all_files:
        stats = analyze_dataset(str(file_path))
        if stats:
            all_stats.append(stats)
            print_stats(stats)
    if all_stats:
        print(f"\n{'='*70}")
        print("汇总统计")
        print(f"{'='*70}")
        print(f"{'数据集':<30} {'总样本':<12} {'正常':<12} {'异常':<12} {'平均长度':<12}")
        print(f"{'-'*70}")
        total_all = 0
        total_normal = 0
        total_anomaly = 0
        for stats in all_stats:
            print(f"{stats['file_name']:<30} "
                  f"{stats['total_samples']:<12,} "
                  f"{stats['normal_samples']:<12,} "
                  f"{stats['anomaly_samples']:<12,} "
                  f"{stats['avg_sentence_length']:<12.2f}")
            total_all += stats['total_samples']
            total_normal += stats['normal_samples']
            total_anomaly += stats['anomaly_samples']
        print(f"{'-'*70}")
        print(f"{'总计':<30} "
              f"{total_all:<12,} "
              f"{total_normal:<12,} "
              f"{total_anomaly:<12,}")
        print(f"\n整体异常率: {total_anomaly/total_all*100:.2f}%")
        excel_path = data_dir / f"dataset_statistics.xlsx"
        save_to_excel(all_stats, excel_path)
    print(f"\n{'='*70}")
    print("分析完成!")
    print(f"{'='*70}")
if __name__ == "__main__":
    main()
