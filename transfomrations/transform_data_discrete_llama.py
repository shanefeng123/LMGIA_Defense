from transformers import LlamaForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig
from utils import *
import json
import torch
import random
from argparse import ArgumentParser

parser = ArgumentParser("Transform data using discrete optimization")
parser.add_argument("--data", type=str, default="enron", help="The dataset to use")
parser.add_argument("--beam_size", type=int, default=1, help="The beam size to use")
parser.add_argument("--device", type=str, default="cuda:0", help="The device to use")
parser.add_argument("--num_of_samples", type=int, default=1000, help="The number of samples to use")
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="The model to use")
args = parser.parse_args()

random.seed(42)
torch.manual_seed(42)

data = args.data
beam_size = args.beam_size
device = args.device
num_of_samples = args.num_of_samples
model_name = args.model_name

# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     bnb_8bit_use_double_quant=True,
#     bnb_8bit_quant_type="nf8",
#     bnb_8bit_compute_dtype=torch.bfloat16,
# )

token = "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"

model = LlamaForCausalLM.from_pretrained(model_name, token=token,
                                         device_map=device)
tokenizer = LlamaTokenizerFast.from_pretrained(model_name, token=token, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

dataset = json.load(open(f"../data/{data}.json", "r"))
# Randomly sample 1000 samples
sentences = random.choices(dataset, k=num_of_samples)

embedding_layer = model.model.embed_tokens
embeddings = model.model.embed_tokens.weight.clone().detach()
k = 10

nearest_neighbors = {}
for i in range(len(embeddings)):
    if i not in tokenizer.all_special_ids:
        if i % 100 == 0:
            print("Processing token", i + 1, "out of", len(embeddings) - len(tokenizer.all_special_ids))
        token_embedding = embeddings[i]
        cosine_similarities = torch.nn.functional.cosine_similarity(embeddings, token_embedding.unsqueeze(0), dim=1)
        sorted_indices = torch.argsort(cosine_similarities, descending=True)
        nearest_neighbors[i] = sorted_indices[1:k + 1].tolist()

tokenized_sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
tokenized_sentences["labels"] = tokenized_sentences["input_ids"].clone()
dataset = MyDataset(tokenized_sentences)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

transformed_data = []
i = 0
for batch in dataloader:
    print("Processing batch ", i + 1, "out of", len(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    batch_embeddings = embedding_layer(input_ids)
    with torch.no_grad():
        batch_outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        batch_hidden_states = batch_outputs.hidden_states

    optimizer = BeamSearchOptimizer(model, batch_hidden_states, batch, device=device,
                                    neighbours=nearest_neighbors, tokenizer=tokenizer, embedding_layer=embedding_layer,
                                    beam_size=beam_size,
                                    iteration=10,
                                    alpha=0.7, beta=0.3)

    optimizer.initialize()
    dummy_data, loss = optimizer.optimize()
    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    print(tokenizer.decode(dummy_data[0], skip_special_tokens=True))
    transformed_data.append(tokenizer.decode(dummy_data[0], skip_special_tokens=True))
    i += 1
#
with open(f"../data/{model_name.split('/')[-1]}/{num_of_samples}_transformed_{data}_beam_{beam_size}_data.json",
          "w") as f:
    json.dump({"transformed_sentences": transformed_data, "original_sentences": sentences}, f)
