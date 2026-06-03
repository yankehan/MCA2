from openai import OpenAI
import tiktoken
import time
import random
import numpy as np
import os
import argparse
import traceback
parser = argparse.ArgumentParser(description='Generate OpenAI Large embeddings for text datasets')
parser.add_argument('--dataset', type=str, default='email_spam',
                    help='Dataset name (e.g., email_spam, smsspam, covid_fake, liar2, hate_speech, olid)')
parser.add_argument('--resume', action='store_true', default=True, help='Resume from existing batch files (default: True)')
args = parser.parse_args()
dataset = args.dataset
data_dir = '../data/'+dataset + '.npz'
loaded = np.load(data_dir, allow_pickle=True)
texts = loaded['data'].tolist()
labels = loaded['label'].tolist()
start_time = time.time()
client = OpenAI(
    base_url= "https://api.chatanywhere.tech/v1",
    api_key = 'xxx'
)
tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")
MAX_TOKENS_PER_TEXT = 8191
MAX_BATCH_SIZE = 2048
MAX_BATCH_TOKENS = 250000
TARGET_BATCH_SIZE = 1000
print(f"Total texts to process: {len(texts)}")
print(f"Target batch size: {TARGET_BATCH_SIZE}")
print(f"Max batch tokens: {MAX_BATCH_TOKENS}")
batches = []
batch = []
batch_token_count = 0
for text in texts:
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    if token_count > MAX_TOKENS_PER_TEXT:
        print(f"Text is too long ({token_count} tokens). It will be truncated.")
        text = tokenizer.decode(tokens[:MAX_TOKENS_PER_TEXT])
        token_count = MAX_TOKENS_PER_TEXT
    should_create_new_batch = (
        len(batch) >= TARGET_BATCH_SIZE or
        batch_token_count + token_count > MAX_BATCH_TOKENS
    )
    if should_create_new_batch and batch:
        batches.append(batch)
        print(f"  Batch {len(batches)} created: {len(batch)} texts, ~{batch_token_count} tokens")
        batch = []
        batch_token_count = 0
    batch.append(text)
    batch_token_count += token_count
if batch:
    batches.append(batch)
    print(f"  Batch {len(batches)} created: {len(batch)} texts, ~{batch_token_count} tokens")
print(f"Total batches created: {len(batches)}")
batch_dir = os.path.join(dataset, "openai_large_batches")
if not os.path.exists(batch_dir):
    os.makedirs(batch_dir)
MODEL_NAME = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIM = 3072
def _sleep_seconds_for_retry(attempt: int, base: float = 5.0, cap: float = 120.0) -> float:
    exp = min(cap, base * (2 ** attempt))
    return exp + random.uniform(0.0, 1.0)
def _request_embeddings(text_list, max_retries: int = 8):
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=text_list,
                model=MODEL_NAME,
            )
            return np.array([item.embedding for item in response.data], dtype=np.float32)
        except Exception as e:
            last_exc = e
            wait_s = _sleep_seconds_for_retry(attempt)
            print(
                f"\n  Request failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}. Retrying in {wait_s:.1f}s"
            )
            time.sleep(wait_s)
    raise last_exc
def _embed_with_fallback(text_list, max_retries: int = 8):
    try:
        return _request_embeddings(text_list, max_retries=max_retries)
    except Exception as e:
        if len(text_list) <= 1:
            print(f"\n  WARNING: Embedding failed for 1 text after all retries: {type(e).__name__}: {e}. Filling with NaN.")
            return np.full((len(text_list), DEFAULT_EMBEDDING_DIM), np.nan, dtype=np.float32)
        print(f"\n  Batch of {len(text_list)} failed ({type(e).__name__}), splitting in half and retrying...")
        mid = len(text_list) // 2
        left = _embed_with_fallback(text_list[:mid], max_retries=max_retries)
        right = _embed_with_fallback(text_list[mid:], max_retries=max_retries)
        return np.vstack([left, right])
for i, batch in enumerate(batches):
    batch_file = os.path.join(batch_dir, f"batch_{i:05d}.npy")
    if args.resume and os.path.exists(batch_file):
        print(f"Processing batch {i + 1}/{len(batches)} with {len(batch)} texts...", end = "\t")
        print(f"  Batch {i + 1} skipped (exists)")
        continue
    print(f"Processing batch {i + 1}/{len(batches)} with {len(batch)} texts...", end = "\t")
    batch_start_time = time.time()
    try:
        batch_embeddings = _embed_with_fallback(batch, max_retries=8)
    except Exception as e:
        err_file = os.path.join(batch_dir, f"batch_{i:05d}.error.txt")
        with open(err_file, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"\n  Batch {i + 1} failed: {type(e).__name__}: {e} (saved: {err_file})")
        batch_embeddings = np.full((len(batch), DEFAULT_EMBEDDING_DIM), np.nan, dtype=np.float32)
    np.save(batch_file, batch_embeddings)
    batch_time = time.time() - batch_start_time
    print(f"  Batch {i + 1} completed in {batch_time:.2f} seconds")
end_time = time.time()
save_path = dataset + "/openai_large_"+ dataset +".npy"
folder_path = os.path.dirname(save_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
print("Merging batch files...")
first_batch = True
for i in range(len(batches)):
    bf = os.path.join(batch_dir, f"batch_{i:05d}.npy")
    if not os.path.exists(bf):
        print(f"Warning: Missing batch file: {bf}. It will be skipped during merge.")
        continue
    batch_data = np.load(bf, allow_pickle=True)
    if first_batch:
        embeddings = batch_data
        first_batch = False
    else:
        embeddings = np.vstack([embeddings, batch_data])
    if (i + 1) % 10 == 0 or i == len(batches) - 1:
        print(f"  Merged {i + 1}/{len(batches)} batches")
nan_count = np.isnan(embeddings).any(axis=1).sum()
if nan_count > 0:
    print(f"WARNING: {nan_count}/{embeddings.shape[0]} embeddings contain NaN values (failed API calls).")
np.save(save_path, embeddings)
print(save_path,"successful!")
