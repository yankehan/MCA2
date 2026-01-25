import torch
from sentence_transformers import SentenceTransformer
import time
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='Generate Stella embeddings for text datasets')
parser.add_argument('--dataset', type=str, default='hate_speech',
                    help='Dataset name (e.g., email_spam, smsspam, covid_fake, liar2, hate_speech, olid)')
args = parser.parse_args()
dataset = args.dataset
data_dir = '../data/'+dataset + '.npz'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded = np.load(data_dir, allow_pickle=True)
texts = loaded['data'].tolist()
labels = loaded['label'].tolist()
model_name = "dunzhang/stella_en_400M_v5"
cache_dir = "model/stella_en_400M_v5"
model = SentenceTransformer("dunzhang/stella_en_400M_v5", cache_folder = "model/stella_en_400M_v5", trust_remote_code=True).cuda()
start_time = time.time()
print(f"Starting Stella embedding generation for {len(texts)} texts")
all_embeddings = model.encode(texts, show_progress_bar=True)
end_time = time.time()
elapsed_time = end_time - start_time
save_path = dataset + "/stella_"+ dataset +".npy"
folder_path = os.path.dirname(save_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
np.save(save_path, all_embeddings)
print(f"Saving embeddings to {save_path}") 
