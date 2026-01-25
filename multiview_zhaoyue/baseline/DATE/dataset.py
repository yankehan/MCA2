import json
import os
from typing import List, Tuple
def read_jsonl_texts(path: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(str(obj.get("text", "")))
            labels.append(int(obj.get("label", 0)))
    return texts, labels
def maybe_generate_jsonl_from_npz(
    dataset_name: str,
    data_root: str,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> None:
    train_jsonl = os.path.join(data_root, f"{dataset_name}_train_data.jsonl")
    test_jsonl = os.path.join(data_root, f"{dataset_name}_test_data.jsonl")
    if os.path.exists(train_jsonl) and os.path.exists(test_jsonl):
        return
    npz_path = os.path.join(data_root, f"{dataset_name}.npz")
    if not os.path.exists(npz_path):
        return
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    texts = data["data"]
    labels = data["label"]
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    if len(normal_idx) == 0:
        raise ValueError(f"No normal samples (label=0) found in {npz_path}")
    if len(anomaly_idx) == 0:
        raise ValueError(f"No anomaly samples (label=1) found in {npz_path}")
    rng = np.random.RandomState(seed)
    rng.shuffle(normal_idx)
    n_train = int(len(normal_idx) * train_ratio)
    train_normal_idx = normal_idx[:n_train]
    test_normal_idx = normal_idx[n_train:]
    train_texts = [str(texts[i]) for i in train_normal_idx]
    train_labels = [0 for _ in train_normal_idx]
    test_idx = np.concatenate([test_normal_idx, anomaly_idx])
    rng.shuffle(test_idx)
    test_texts = [str(texts[i]) for i in test_idx]
    test_labels = [int(labels[i]) for i in test_idx]
    def dump_jsonl(path: str, xs: List[str], ys: List[int]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for x, y in zip(xs, ys):
                f.write(json.dumps({"text": x, "label": int(y)}, ensure_ascii=False) + "\n")
    dump_jsonl(train_jsonl, train_texts, train_labels)
    dump_jsonl(test_jsonl, test_texts, test_labels)
def ensure_date_txts(
    dataset_name: str,
    data_root: str,
    work_dir: str,
    seed: int = 42,
) -> Tuple[str, str, str]:
    maybe_generate_jsonl_from_npz(dataset_name=dataset_name, data_root=data_root, seed=seed)
    train_jsonl = os.path.join(data_root, f"{dataset_name}_train_data.jsonl")
    test_jsonl = os.path.join(data_root, f"{dataset_name}_test_data.jsonl")
    if not os.path.exists(train_jsonl) or not os.path.exists(test_jsonl):
        raise FileNotFoundError(
            f"缺少数据文件: {dataset_name}\n"
            f"需要存在: {train_jsonl} 和 {test_jsonl}\n"
            f"或者存在: {os.path.join(data_root, dataset_name + '.npz')} 以便自动生成 jsonl"
        )
    os.makedirs(work_dir, exist_ok=True)
    train_txt = os.path.join(work_dir, "train.txt")
    test_txt = os.path.join(work_dir, "test.txt")
    outliers_txt = os.path.join(work_dir, "outliers.txt")
    train_texts, train_labels = read_jsonl_texts(train_jsonl)
    if any(l != 0 for l in train_labels):
        raise ValueError(
            f"训练集必须全部为正常样本(label=0)，但 {train_jsonl} 中出现了: {sorted(set(train_labels))}"
        )
    test_texts, test_labels = read_jsonl_texts(test_jsonl)
    test_inliers = [t for t, y in zip(test_texts, test_labels) if y == 0]
    test_outliers = [t for t, y in zip(test_texts, test_labels) if y == 1]
    if len(test_outliers) == 0:
        raise ValueError(f"测试集必须包含异常样本(label=1)，但 {test_jsonl} 中没有")
    def dump_txt(lines: List[str], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for t in lines:
                t = (t or "").replace("\n", " ").strip()
                if not t:
                    continue
                f.write(t + "\n")
    dump_txt(train_texts, train_txt)
    dump_txt(test_inliers, test_txt)
    dump_txt(test_outliers, outliers_txt)
    return train_txt, test_txt, outliers_txt
