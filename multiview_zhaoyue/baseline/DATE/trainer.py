import os
import time
from typing import List, Optional, Tuple
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(int(seed))
    try:
        import numpy as np
        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        return
def set_seed_42() -> None:
    set_seed(42)
def _read_txt_lines(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                lines.append(t)
    return lines
def _make_mlm_inputs(
    input_ids,
    attention_mask,
    tokenizer,
    pseudo_mask,
):
    import torch
    bsz, seqlen = input_ids.shape
    labels = input_ids.clone()
    special_ids = set(
        x
        for x in [
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        ]
        if x is not None
    )
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("tokenizer 缺少 mask_token_id，无法进行 MLM")
    pm_len = len(pseudo_mask)
    mask_matrix = torch.zeros((bsz, seqlen), dtype=torch.bool, device=input_ids.device)
    for i in range(bsz):
        max_pos = min(seqlen - 1, pm_len + 1)
        if max_pos <= 1:
            continue
        row_mask = torch.tensor(pseudo_mask[: max_pos - 1], device=input_ids.device, dtype=torch.bool)
        mask_matrix[i, 1:max_pos] = row_mask
    mask_matrix &= attention_mask.bool()
    for sid in special_ids:
        mask_matrix &= input_ids.ne(int(sid))
    labels[~mask_matrix] = -100
    masked_input = input_ids.clone()
    masked_input[mask_matrix] = int(mask_token_id)
    return masked_input, labels
def run_date_one_dataset(
    dataset_name: str,
    data_root: str,
    out_dir: str,
    device: str = "cuda",
    seed: int = 42,
    seq_len: int = 128,
    num_train_epochs: int = 20,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    anomaly_batch_size: int = 16,
    max_lr: float = 1e-5,
    min_lr: float = 1e-4,
    warmup_steps: int = 1000,
    weight_decay: float = 0.1,
    disc_drop: float = 0.5,
    disc_hid_layers: int = 4,
    disc_hid_size: int = 256,
    gen_hid_layers: int = 1,
    gen_hid_size: int = 16,
    rtd_loss_weight: int = 50,
    rmd_loss_weight: int = 100,
    mlm_loss_weight: int = 1,
    log_every_n_epochs: int = 5,
    suppress_internal_output: bool = True,
) -> Tuple[float, float, float]:
    import numpy as np
    import torch
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score, average_precision_score
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from .dataset import ensure_date_txts
    from .model import DateConfig, DateModel, compute_rtd_anomaly_score
    work_dir = os.path.join(out_dir, "_work", dataset_name)
    train_txt, test_txt, outliers_txt = ensure_date_txts(
        dataset_name=dataset_name,
        data_root=data_root,
        work_dir=work_dir,
        seed=int(seed),
    )
    set_seed(int(seed))
    use_cuda = device.lower().startswith("cuda") and torch.cuda.is_available()
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    max_length = int(seq_len) + 2
    cfg = DateConfig(
        seq_len=int(seq_len),
        n_masks=50,
        vocab_size=int(getattr(tokenizer, "vocab_size", 30522)),
        gen_hidden_size=int(gen_hid_size),
        gen_num_layers=int(gen_hid_layers),
        disc_hidden_size=int(disc_hid_size),
        disc_num_layers=int(disc_hid_layers),
        dropout=float(disc_drop),
    )
    model = DateModel(cfg=cfg, random_generator=True, seed=int(seed)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(max_lr), weight_decay=float(weight_decay))
    train_texts = _read_txt_lines(train_txt)
    test_inlier_texts = _read_txt_lines(test_txt)
    test_outlier_texts = _read_txt_lines(outliers_txt)
    def _collate_texts(batch: List[str]):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids")
    train_loader = DataLoader(train_texts, batch_size=int(train_batch_size), shuffle=True, num_workers=0, collate_fn=_collate_texts)
    eval_in_loader = DataLoader(test_inlier_texts, batch_size=int(eval_batch_size), shuffle=False, num_workers=0, collate_fn=_collate_texts)
    eval_out_loader = DataLoader(test_outlier_texts, batch_size=int(eval_batch_size), shuffle=False, num_workers=0, collate_fn=_collate_texts)
    t0 = time.time()
    model.train()
    rng = np.random.RandomState(int(seed))
    def _eval_current_model_auroc() -> float:
        model.eval()
        in_scores = []
        out_scores = []
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids in eval_in_loader:
                input_ids = input_ids.to(torch_device)
                attention_mask = attention_mask.to(torch_device)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(torch_device)
                s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
                in_scores.append(s.detach().cpu().numpy())
            for input_ids, attention_mask, token_type_ids in eval_out_loader:
                input_ids = input_ids.to(torch_device)
                attention_mask = attention_mask.to(torch_device)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(torch_device)
                s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
                out_scores.append(s.detach().cpu().numpy())
        in_scores_np = np.concatenate(in_scores, axis=0) if len(in_scores) else np.array([])
        out_scores_np = np.concatenate(out_scores, axis=0) if len(out_scores) else np.array([])
        y_true = np.concatenate([np.zeros_like(in_scores_np), np.ones_like(out_scores_np)], axis=0)
        y_score = np.concatenate([in_scores_np, out_scores_np], axis=0)
        return float(roc_auc_score(y_true, y_score))
    def _eval_current_model_auprc() -> float:
        model.eval()
        in_scores = []
        out_scores = []
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids in eval_in_loader:
                input_ids = input_ids.to(torch_device)
                attention_mask = attention_mask.to(torch_device)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(torch_device)
                s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
                in_scores.append(s.detach().cpu().numpy())
            for input_ids, attention_mask, token_type_ids in eval_out_loader:
                input_ids = input_ids.to(torch_device)
                attention_mask = attention_mask.to(torch_device)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(torch_device)
                s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
                out_scores.append(s.detach().cpu().numpy())
        in_scores_np = np.concatenate(in_scores, axis=0) if len(in_scores) else np.array([])
        out_scores_np = np.concatenate(out_scores, axis=0) if len(out_scores) else np.array([])
        y_true = np.concatenate([np.zeros_like(in_scores_np), np.ones_like(out_scores_np)], axis=0)
        y_score = np.concatenate([in_scores_np, out_scores_np], axis=0)
        return float(average_precision_score(y_true, y_score))
    for _epoch in range(int(num_train_epochs)):
        if suppress_internal_output:
            train_iter = train_loader
        else:
            train_iter = tqdm(
                train_loader,
                desc=f"{dataset_name} train {(_epoch + 1)}/{int(num_train_epochs)}",
                leave=False,
                position=1,
                dynamic_ncols=True,
            )
        for _step, (input_ids, attention_mask, token_type_ids) in enumerate(train_iter):
            input_ids = input_ids.to(torch_device)
            attention_mask = attention_mask.to(torch_device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(torch_device)
            mask_idx = int(rng.randint(0, len(model.masks)))
            pseudo_mask = model.masks[mask_idx]
            masked_input, mlm_labels = _make_mlm_inputs(input_ids, attention_mask, tokenizer, pseudo_mask)
            rmd_labels = torch.full((input_ids.size(0),), mask_idx, device=torch_device, dtype=torch.long)
            outputs = model(
                input_ids=masked_input,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                mlm_labels=mlm_labels,
                rmd_labels=rmd_labels,
                replace_tokens=True,
            )
            loss = 0.0
            if outputs.get("g_loss") is not None:
                loss = loss + float(mlm_loss_weight) * outputs["g_loss"]
            if outputs.get("rtd_loss") is not None:
                loss = loss + float(rtd_loss_weight) * outputs["rtd_loss"]
            if outputs.get("rmd_loss") is not None:
                loss = loss + float(rmd_loss_weight) * outputs["rmd_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if not suppress_internal_output and hasattr(train_iter, "set_postfix"):
                if _step % 10 == 0:
                    try:
                        train_iter.set_postfix(loss=float(loss.detach().cpu()))
                    except Exception:
                        pass
        if int(log_every_n_epochs) > 0 and ((_epoch + 1) % int(log_every_n_epochs) == 0):
            auroc_mid = _eval_current_model_auroc()
            auprc_mid = _eval_current_model_auprc()
            model.train()
            elapsed = float(time.time() - t0)
            msg = f"[DATE] dataset={dataset_name} epoch={_epoch + 1}/{int(num_train_epochs)} AUROC={auroc_mid:.6f} AUPRC={auprc_mid:.6f} elapsed_sec={elapsed:.2f}"
            if suppress_internal_output:
                print(msg)
            else:
                try:
                    tqdm.write(msg)
                except Exception:
                    print(msg)
    model.eval()
    in_scores = []
    out_scores = []
    with torch.no_grad():
        if suppress_internal_output:
            eval_in_iter = eval_in_loader
        else:
            eval_in_iter = tqdm(
                eval_in_loader,
                desc=f"{dataset_name} eval inliers",
                leave=False,
                position=1,
                dynamic_ncols=True,
            )
        for input_ids, attention_mask, token_type_ids in eval_in_iter:
            input_ids = input_ids.to(torch_device)
            attention_mask = attention_mask.to(torch_device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(torch_device)
            s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
            in_scores.append(s.detach().cpu().numpy())
        if suppress_internal_output:
            eval_out_iter = eval_out_loader
        else:
            eval_out_iter = tqdm(
                eval_out_loader,
                desc=f"{dataset_name} eval outliers",
                leave=False,
                position=1,
                dynamic_ncols=True,
            )
        for input_ids, attention_mask, token_type_ids in eval_out_iter:
            input_ids = input_ids.to(torch_device)
            attention_mask = attention_mask.to(torch_device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(torch_device)
            s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
            out_scores.append(s.detach().cpu().numpy())
    in_scores_np = np.concatenate(in_scores, axis=0) if len(in_scores) else np.array([])
    out_scores_np = np.concatenate(out_scores, axis=0) if len(out_scores) else np.array([])
    y_true = np.concatenate([np.zeros_like(in_scores_np), np.ones_like(out_scores_np)], axis=0)
    y_score = np.concatenate([in_scores_np, out_scores_np], axis=0)
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    t1 = time.time()
    return auroc, auprc, float(t1 - t0)
def run_date_one_dataset_efficiency(
    dataset_name: str,
    data_root: str,
    out_dir: str,
    device: str = "cuda",
    seed: int = 42,
    seq_len: int = 128,
    num_train_epochs: int = 20,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    anomaly_batch_size: int = 16,
    max_lr: float = 1e-5,
    min_lr: float = 1e-4,
    warmup_steps: int = 1000,
    weight_decay: float = 0.1,
    disc_drop: float = 0.5,
    disc_hid_layers: int = 4,
    disc_hid_size: int = 256,
    gen_hid_layers: int = 1,
    gen_hid_size: int = 16,
    rtd_loss_weight: int = 50,
    rmd_loss_weight: int = 100,
    mlm_loss_weight: int = 1,
    log_every_n_epochs: int = 0,
    suppress_internal_output: bool = True,
) -> Tuple[float, float, float, float]:
    import numpy as np
    import torch
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from .dataset import ensure_date_txts
    from .model import DateConfig, DateModel, compute_rtd_anomaly_score
    total_t0 = time.time()
    work_dir = os.path.join(out_dir, "_work", dataset_name)
    train_txt, test_txt, outliers_txt = ensure_date_txts(
        dataset_name=dataset_name,
        data_root=data_root,
        work_dir=work_dir,
        seed=int(seed),
    )
    set_seed(int(seed))
    use_cuda = device.lower().startswith("cuda") and torch.cuda.is_available()
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    max_length = int(seq_len) + 2
    cfg = DateConfig(
        seq_len=int(seq_len),
        n_masks=50,
        vocab_size=int(getattr(tokenizer, "vocab_size", 30522)),
        gen_hidden_size=int(gen_hid_size),
        gen_num_layers=int(gen_hid_layers),
        disc_hidden_size=int(disc_hid_size),
        disc_num_layers=int(disc_hid_layers),
        dropout=float(disc_drop),
    )
    model = DateModel(cfg=cfg, random_generator=True, seed=int(seed)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(max_lr), weight_decay=float(weight_decay))
    train_texts = _read_txt_lines(train_txt)
    test_inlier_texts = _read_txt_lines(test_txt)
    test_outlier_texts = _read_txt_lines(outliers_txt)
    def _collate_texts(batch: List[str]):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids")
    train_loader = DataLoader(train_texts, batch_size=int(train_batch_size), shuffle=True, num_workers=0, collate_fn=_collate_texts)
    eval_in_loader = DataLoader(test_inlier_texts, batch_size=int(eval_batch_size), shuffle=False, num_workers=0, collate_fn=_collate_texts)
    eval_out_loader = DataLoader(test_outlier_texts, batch_size=int(eval_batch_size), shuffle=False, num_workers=0, collate_fn=_collate_texts)
    model.train()
    rng = np.random.RandomState(int(seed))
    train_t0 = time.time()
    for _epoch in range(int(num_train_epochs)):
        if suppress_internal_output:
            train_iter = train_loader
        else:
            train_iter = tqdm(
                train_loader,
                desc=f"{dataset_name} train {(_epoch + 1)}/{int(num_train_epochs)}",
                leave=False,
                position=1,
                dynamic_ncols=True,
            )
        for _step, (input_ids, attention_mask, token_type_ids) in enumerate(train_iter):
            input_ids = input_ids.to(torch_device)
            attention_mask = attention_mask.to(torch_device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(torch_device)
            mask_idx = int(rng.randint(0, len(model.masks)))
            pseudo_mask = model.masks[mask_idx]
            masked_input, mlm_labels = _make_mlm_inputs(input_ids, attention_mask, tokenizer, pseudo_mask)
            rmd_labels = torch.full((input_ids.size(0),), mask_idx, device=torch_device, dtype=torch.long)
            outputs = model(
                input_ids=masked_input,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                mlm_labels=mlm_labels,
                rmd_labels=rmd_labels,
                replace_tokens=True,
            )
            loss = 0.0
            if outputs.get("g_loss") is not None:
                loss = loss + float(mlm_loss_weight) * outputs["g_loss"]
            if outputs.get("rtd_loss") is not None:
                loss = loss + float(rtd_loss_weight) * outputs["rtd_loss"]
            if outputs.get("rmd_loss") is not None:
                loss = loss + float(rmd_loss_weight) * outputs["rmd_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if not suppress_internal_output and hasattr(train_iter, "set_postfix"):
                if _step % 10 == 0:
                    try:
                        train_iter.set_postfix(loss=float(loss.detach().cpu()))
                    except Exception:
                        pass
        if int(log_every_n_epochs) > 0 and ((_epoch + 1) % int(log_every_n_epochs) == 0):
            model.eval()
            with torch.no_grad():
                _ = 0
                for input_ids, attention_mask, token_type_ids in eval_in_loader:
                    input_ids = input_ids.to(torch_device)
                    attention_mask = attention_mask.to(torch_device)
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(torch_device)
                    _ = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
            model.train()
    train_t1 = time.time()
    train_time_sec = float(train_t1 - train_t0)
    test_t0 = time.time()
    model.eval()
    in_scores = []
    out_scores = []
    with torch.no_grad():
        if suppress_internal_output:
            eval_in_iter = eval_in_loader
        else:
            eval_in_iter = tqdm(
                eval_in_loader,
                desc=f"{dataset_name} eval inliers",
                leave=False,
                position=1,
                dynamic_ncols=True,
            )
        for input_ids, attention_mask, token_type_ids in eval_in_iter:
            input_ids = input_ids.to(torch_device)
            attention_mask = attention_mask.to(torch_device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(torch_device)
            s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
            in_scores.append(s.detach().cpu().numpy())
        if suppress_internal_output:
            eval_out_iter = eval_out_loader
        else:
            eval_out_iter = tqdm(
                eval_out_loader,
                desc=f"{dataset_name} eval outliers",
                leave=False,
                position=1,
                dynamic_ncols=True,
            )
        for input_ids, attention_mask, token_type_ids in eval_out_iter:
            input_ids = input_ids.to(torch_device)
            attention_mask = attention_mask.to(torch_device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(torch_device)
            s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
            out_scores.append(s.detach().cpu().numpy())
    in_scores_np = np.concatenate(in_scores, axis=0) if len(in_scores) else np.array([])
    out_scores_np = np.concatenate(out_scores, axis=0) if len(out_scores) else np.array([])
    y_true = np.concatenate([np.zeros_like(in_scores_np), np.ones_like(out_scores_np)], axis=0)
    y_score = np.concatenate([in_scores_np, out_scores_np], axis=0)
    auroc = float(roc_auc_score(y_true, y_score))
    test_t1 = time.time()
    test_time_sec = float(test_t1 - test_t0)
    total_time_sec = float(time.time() - total_t0)
    return auroc, train_time_sec, test_time_sec, total_time_sec
