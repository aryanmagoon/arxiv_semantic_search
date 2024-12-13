import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm
def freeze_all_except_last_xformer_block(model):
    
    try:
        transformer = model[0].auto_model.transformer
    except AttributeError:
        transformer = model.transformer
        
    last_layer_idx = len(transformer.layer) - 1
    
    for name, param in model.named_parameters():
        
        param.requires_grad = False
        
        if f"transformer.layer.{last_layer_idx}" in name:
            param.requires_grad = True

def print_trainable_params(model):
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    print("------Trainable Parameters--------")
    for name in trainable_params:
        print(name)
    print('total trainable params', len(trainable_params))

train_df = pd.read_csv('train.csv')

qs_train = train_df['query']
abstracts_train = train_df['abstract']
class QueryAbstractDataset(Dataset):
  def __init__(self, queries, abstracts):
    self.queries = queries
    self.abstracts = abstracts

  def __len__(self):
    return len(self.queries)
  def __getitem__(self, idx):
    query = self.queries[idx]
    abstract = self.abstracts[idx]
    return query, abstract

dataset = QueryAbstractDataset(qs_train, abstracts_train)
batch_size = 16

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
def adjust_dropout_rate(model, dropout_rate):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate
    return model
query_encoder = SentenceTransformer('multi-qa-distilbert-dot-v1')
freeze_all_except_last_xformer_block(query_encoder)
abstract_encoder = SentenceTransformer('multi-qa-distilbert-dot-v1')
freeze_all_except_last_xformer_block(abstract_encoder)

num_epochs = 10
learning_rate = 5e-6
weight_decay = 0.1
dropout = 0.3
query_encoder = adjust_dropout_rate(query_encoder, dropout)
abstract_encoder = adjust_dropout_rate(abstract_encoder, dropout)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
query_encoder.to(device)
abstract_encoder.to(device)
query_encoder_trainable_params = [
    param for name, param in query_encoder.named_parameters() 
    if param.requires_grad
]

abstract_encoder_trainable_params = [param for name, param in abstract_encoder.named_parameters() if param.requires_grad]

trainable_params = query_encoder_trainable_params + abstract_encoder_trainable_params
optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay = weight_decay)
criterion = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    query_encoder.train()
    abstract_encoder.train()
    total_loss = 0
    for queries_batch, abstracts_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        query_inputs = query_encoder.tokenize(queries_batch)
        for key in query_inputs:
            query_inputs[key] = query_inputs[key].to(device)
        query_embeddings = query_encoder(query_inputs)['sentence_embedding']

        abstract_inputs = abstract_encoder.tokenize(abstracts_batch)
        for key in abstract_inputs:
            abstract_inputs[key] = abstract_inputs[key].to(device)
        abstract_embeddings = abstract_encoder(abstract_inputs)['sentence_embedding']

        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        abstract_embeddings = F.normalize(abstract_embeddings, p=2, dim=1)

        scores = torch.matmul(query_embeddings, abstract_embeddings.T)

        labels = torch.arange(scores.size(0)).to(device)

        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
query_encoder.save('query_encoder_model_frozen_2_final')
abstract_encoder.save('abstract_encoder_model_frozen_2_final')