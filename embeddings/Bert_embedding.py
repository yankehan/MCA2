from transformers import BertTokenizer, BertModel
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Generate BERT embeddings for text datasets')
parser.add_argument('--dataset', type=str, default='hate_speech',
                    help='Dataset name (e.g., email_spam, smsspam, covid_fake, liar2, hate_speech, olid)')
args = parser.parse_args()
dataset = args.dataset
data_dir = '../data/'+dataset + '.npz'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded = np.load(data_dir, allow_pickle=True)
texts = loaded['data'].tolist()
labels = loaded['label'].tolist()
model_name = "bert-base-uncased"
cache_dir = "model/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
model = model.to(device)
start_time = time.time()
batch_size = 16
sentence_embeddings = []
dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
print(f"Starting BERT embedding generation for {len(texts)} texts, batch_size={batch_size}")
for batch in tqdm(dataloader, desc="BERT Embedding", unit="batch"):
    encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    sentence_embeddings.append(batch_embeddings)
sentence_embeddings = np.vstack(sentence_embeddings)
end_time = time.time()
elapsed_time = end_time - start_time
save_path = dataset + "/bert_"+ dataset +".npy"
folder_path = os.path.dirname(save_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
np.save(save_path, sentence_embeddings)
print(f"Saving embeddings to {save_path}") 
