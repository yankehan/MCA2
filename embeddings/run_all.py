import subprocess
import os
import sys
import argparse
from pathlib import Path
current_dir = Path(__file__).parent
embedding_scripts = {
    "Bert_embedding.py": "bert_{dataset}.npy",
    "Llama-3.2-1B_embedding.py": "llama_{dataset}.npy",
    "Qwen2.5_embedding.py": "qwen_{dataset}.npy",
    "all-MiniLM-L6-v2_embedding.py": "minilm_{dataset}.npy",
    "stella_embedding.py": "stella_{dataset}.npy"
}
def check_output_exists(output_pattern, dataset_name):
    output_filename = output_pattern.format(dataset=dataset_name)
    output_path = current_dir / dataset_name / output_filename
    return output_path.exists(), output_path
def run_embedding_script(script_name, output_pattern, dataset_name, skip_existing=True):
    script_path = current_dir / script_name
    if not script_path.exists():
        print(f"⚠️  脚本不存在: {script_name}")
        return False
    exists, output_path = check_output_exists(output_pattern, dataset_name)
    if exists and skip_existing:
        print(f"\n{'='*60}")
        print(f"⏭️  跳过: {script_name}")
        print(f"� 数据集: {dataset_name}")
        print(f"✅ 文件已存在: {output_path}")
        print(f"{'='*60}\n")
        return True
    print(f"\n{'='*60}")
    print(f"�🚀 开始运行: {script_name}")
    print(f"📊 数据集: {dataset_name}")
    print(f"{'='*60}\n")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--dataset", dataset_name],
            cwd=str(current_dir),
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            print(f"\n✅ {script_name} 运行成功!\n")
            return True
        else:
            print(f"\n❌ {script_name} 运行失败 (返回码: {result.returncode})\n")
            return False
    except Exception as e:
        print(f"\n❌ {script_name} 运行出错: {str(e)}\n")
        return False
def main():
    parser = argparse.ArgumentParser(
        description='批量运行所有 embedding 脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_all_embeddings.py --dataset bbc
  python run_all_embeddings.py --dataset email_spam
  python run_all_embeddings.py --dataset hate_speech
支持的数据集:
  email_spam, smsspam, covid_fake, liar2, hate_speech, olid
  agnews, bbc, emotion, movie_review, N24News, yelp_review_polarity
        """
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='bbc',
        help='数据集名称 (默认: bbc)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成，即使文件已存在'
    )
    args = parser.parse_args()
    dataset = args.dataset
    skip_existing = not args.force
    print("\n" + "="*60)
    print("🎯 批量运行所有 Embedding 脚本")
    print("="*60)
    print(f"📁 工作目录: {current_dir}")
    print(f"📊 数据集: {dataset}")
    print(f"📝 脚本数量: {len(embedding_scripts)}")
    print(f"⏭️  跳过已存在: {'是' if skip_existing else '否（强制重新生成）'}")
    print("="*60 + "\n")
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_scripts = []
    for script, output_pattern in embedding_scripts.items():
        exists, _ = check_output_exists(output_pattern, dataset)
        if exists and skip_existing:
            skipped_count += 1
        success = run_embedding_script(script, output_pattern, dataset, skip_existing)
        if success:
            success_count += 1
        else:
            failed_count += 1
            failed_scripts.append(script)
    print("\n" + "="*60)
    print("📊 运行总结")
    print("="*60)
    print(f"✅ 成功: {success_count}/{len(embedding_scripts)}")
    print(f"⏭️  跳过: {skipped_count}/{len(embedding_scripts)}")
    print(f"❌ 失败: {failed_count}/{len(embedding_scripts)}")
    if failed_scripts:
        print("\n失败的脚本:")
        for script in failed_scripts:
            print(f"  - {script}")
    print("="*60 + "\n")
if __name__ == "__main__":
    main()
