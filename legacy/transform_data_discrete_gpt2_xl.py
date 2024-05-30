from transformers import GPT2LMHeadModel, GPT2TokenizerFast
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
args = parser.parse_args()

random.seed(42)
torch.manual_seed(42)

data = args.data
beam_size = args.beam_size
device = args.device
num_of_samples = args.num_of_samples

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
model.resize_token_embeddings(len(tokenizer))
model.eval()

dataset = json.load(open(f"../data/{data}.json", "r"))
# Randomly sample 1000 samples
sentences = random.choices(dataset, k=num_of_samples)

for i in range(len(sentences)):
    sentences[i] = "<|startoftext|>" + sentences[i] + "<|endoftext|>"

embedding_layer = model.transformer.wte
embeddings = model.transformer.wte.weight.clone().detach()
k = 10

# Calculate the k nearest neighbors for each token
# For each token in the vocabulary, we calculate the cosine similarity between it and all the other tokens
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
with open(f"../data/gpt2-xl/{num_of_samples}_transformed_{data}_beam_{beam_size}_data.json", "w") as f:
    json.dump({"transformed_sentences": transformed_data, "original_sentences": sentences}, f)
