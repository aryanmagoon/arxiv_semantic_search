import pandas as pd
import json
from sklearn.model_selection import train_test_split

with open("aggregated_results.jsonl", 'r') as file:
    data = [json.loads(line) for line in file]
abs_df = pd.read_json('test.json', lines = True)
abstracts = abs_df['abstract'][:20000].tolist()
queries_final = []
abstracts_final = []

for abstract, query_list in zip(abstracts, data):
    for query in query_list:
        queries_final.append(query)
        abstracts_final.append(abstract)

qs_train, qs_test, abstracts_train, abstracts_test = train_test_split(queries_final, abstracts_final, test_size=0.1)
train_df = pd.DataFrame({
    'query': qs_train,
    'abstract': abstracts_train
})

test_df = pd.DataFrame({
    'query': qs_test,
    'abstract': abstracts_test
})

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)