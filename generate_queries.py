from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import pandas as pd
import json
import pandas as pd
import os
from tqdm import tqdm  # Import tqdm for progress bars

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = 8192,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

import json

with open ('samples.json', 'r') as f:
  samples = json.load(f)
def format_in_context_examples(samples):
    messages = []
    for sample in samples:
        human_message = {
            "from": "human",
            "value": f"Generate 2 questions that someone would read this paper to find out?\n\nAbstract:\n{sample['abstract']}"
        }
        messages.append(human_message)

        assistant_response = f"{sample['q1']}\n{sample['q2']}"
        assistant_message = {
            "from": "assistant",
            "value": assistant_response
        }
        messages.append(assistant_message)
    return messages
in_context_messages = format_in_context_examples(samples[:50])
data = pd.read_json('test.json', lines = True)
new_abstracts = []
for i in range(22000):
  new_abstracts.append(data['abstract'][i])
new_messages = []
for new_abstract in new_abstracts:
    new_message = {
        "from": "human",
        "value": f"Generate 2 questions that someone would read this paper to find out?\n\nAbstract:\n{new_abstract}"
    }
    new_messages.append(new_message)
messages_list = []
for new_message in new_messages:
    messages = in_context_messages + [new_message]
    messages_list.append(messages)
tokenizer.pad_token = tokenizer.eos_token
batch_size = 32
save_interval = 128
start_val = 0
total_samples = 20000


save_directory = "/path/to/directory"
os.makedirs(save_directory, exist_ok=True)


aggregated_save_path = f"{save_directory}/aggregated_results.jsonl"

with open(aggregated_save_path, "w") as f:
    pass

saved_samples = 0

with tqdm(total=total_samples-start_val, desc="Processing Batches", unit="batch") as pbar:
    for start_idx in range(start_val, total_samples, batch_size):
        batch_end_idx = min(start_idx + batch_size, total_samples)
        batch_messages = messages_list[start_idx:batch_end_idx]

        if not batch_messages:
            pbar.update(batch_size)
            continue

        try:
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                tokenize=True,
                return_tensors="pt",
                padding=True
            ).to("cuda")
            generated_ids = model.generate(input_ids=inputs, max_new_tokens=1024, use_cache=True)

            batch_results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            with open(aggregated_save_path, "a") as f:
                for result in batch_results:
                    parsed_result = result.strip().split('\n')[-2:]
                    json_line = json.dumps(parsed_result)
                    f.write(json_line + "\n")
                    saved_samples += 1
            if saved_samples % save_interval == 0:
                print(f"Saved {saved_samples} results to {aggregated_save_path}")
        except Exception as e:
            print(f"Error processing batch {start_idx}-{batch_end_idx}: {e}")
        pbar.update(batch_size)
print("Processing complete!")