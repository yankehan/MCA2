import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.utils.data import DataLoader
import time
import numpy as np
import os
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Generate Qwen2.5 embeddings for text datasets')
parser.add_argument('--dataset', type=str, default='hate_speech',
                    help='Dataset name (e.g., email_spam, smsspam, covid_fake, liar2, hate_speech, olid)')
args = parser.parse_args()
dataset = args.dataset
data_dir = '../data/'+dataset + '.npz'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded = np.load(data_dir, allow_pickle=True)
texts = loaded['data'].tolist()
labels = loaded['label'].tolist()
model_name = "Qwen/Qwen2.5-1.5B"
cache_dir = "model/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir, device_map="auto")
start_time = time.time()
tokenizer.pad_token = tokenizer.eos_token
batch_size = 16
dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
all_embeddings = []
print(f"Starting Qwen embedding generation for {len(texts)} texts, batch_size={batch_size}")
for batch in tqdm(dataloader, desc="Qwen Embedding", unit="batch"):
    encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    valid_token_embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
    batch_embeddings = valid_token_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    all_embeddings.append(batch_embeddings.cpu().numpy())
all_embeddings = np.vstack(all_embeddings)
end_time = time.time()
elapsed_time = end_time - start_time
save_path = dataset + "/qwen_"+ dataset +".npy"
folder_path = os.path.dirname(save_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
np.save(save_path, all_embeddings)
print(f"Saving embeddings to {save_path}") 
