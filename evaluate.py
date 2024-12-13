import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

query_encoder = SentenceTransformer('path/to/model')
abstract_encoder = SentenceTransformer('path/to/model')
query_encoder.eval()
abstract_encoder.eval()
test_df = pd.read_csv('test.csv')
qs_test = test_df['query'].tolist()
abstracts_test = test_df['abstract'].tolist()
train_df = pd.read_csv('train.csv')
abstracts_final = abstracts_test + train_df['abstract'].tolist()
unique_abstracts = list(dict.fromkeys(abstracts_final))
abstract_to_index = {abs_text: idx for idx, abs_text in enumerate(unique_abstracts)}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with torch.no_grad():
    abstract_embeddings = abstract_encoder.encode(unique_abstracts, convert_to_tensor=True, device=device)
    query_embeddings = query_encoder.encode(qs_test, convert_to_tensor=True, device=device)
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
abstract_embeddings = F.normalize(abstract_embeddings, p=2, dim=1)
top1_count = 0
top5_count = 0
num_queries = len(qs_test)
query_embeddings = query_embeddings.cpu()
abstract_embeddings = abstract_embeddings.cpu()
for i, q_emb in enumerate(tqdm(query_embeddings, desc="Evaluating")):
    correct_abstract = abstracts_test[i]
    if correct_abstract not in abstract_to_index:
        continue
    correct_index = abstract_to_index[correct_abstract]
    sims = torch.matmul(q_emb.unsqueeze(0), abstract_embeddings.T).squeeze(0)
    sorted_indices = torch.argsort(sims, descending=True)
    rank_positions = (sorted_indices == correct_index).nonzero(as_tuple=True)
    if len(rank_positions[0]) == 0:
        continue
    found_rank = rank_positions[0].item()
    if found_rank == 0:
        top1_count += 1
    if found_rank < 5:
        top5_count += 1

top1_accuracy = top1_count / num_queries
top5_accuracy = top5_count / num_queries

print(f"Top-1 Accuracy (over full abstracts): {top1_accuracy:.4f}")
print(f"Top-5 Accuracy (over full abstracts): {top5_accuracy:.4f}")