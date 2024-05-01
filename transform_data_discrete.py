import torch
from transformers import BertModel, BertTokenizerFast
from datasets import load_dataset
from utils import *
from BeamSearchOptimizer import *
import json
import random
from argparse import ArgumentParser

parser = ArgumentParser("Transform data using discrete optimization")
parser.add_argument("--data", type=str, default="cola", help="The dataset to use")
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
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)
model.eval()
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
if data == "cola" or data == "sst2":
    dataset = load_dataset("glue", data)["train"]
    sentences = dataset["sentence"]
    labels = dataset["label"]
else:
    dataset = load_dataset(data)["train"]
    sentences = dataset["text"]
    labels = dataset["label"]

# Shuffle the sentences and labels together
shuffled_indices = list(range(len(sentences)))
random.shuffle(shuffled_indices)

# Get 500 positive and 500 negative sentences
positive_indices = [i for i in range(len(labels)) if labels[i] == 1]
negative_indices = [i for i in range(len(labels)) if labels[i] == 0]
shuffled_indices = random.choices(positive_indices, k=int(num_of_samples / 2)) + random.choices(negative_indices, k=int(
    num_of_samples / 2))
sentences = [sentences[i] for i in shuffled_indices]
labels = [labels[i] for i in shuffled_indices]

embedding_layer = model.embeddings.word_embeddings
embeddings = model.embeddings.word_embeddings.weight.clone().detach()
k = 10
# Calculate the k nearest neighbors for each token
# For each token in the vocabulary, we calculate the cosine similarity between it and all the other tokens
# We then sort the tokens based on the cosine similarity and get the top k tokens
# We store the top k tokens in a dictionary
# The key is the token id and the value is a list of the top k tokens
nearest_neighbors = {}
for i in range(len(embeddings)):
    if i not in tokenizer.all_special_ids:
        if i % 100 == 0:
            print("Processing token", i + 1, "out of", len(embeddings) - len(tokenizer.all_special_ids))
        token_embedding = embeddings[i]
        cosine_similarities = torch.nn.functional.cosine_similarity(embeddings, token_embedding.unsqueeze(0), dim=1)
        sorted_indices = torch.argsort(cosine_similarities, descending=True)
        nearest_neighbors[i] = sorted_indices[1:k + 1].tolist()

# # Now use faiss to find the nearest neighbors
# embedding_numpy = embeddings.cpu().numpy()
# faiss.normalize_L2(embedding_numpy)
# index = faiss.IndexFlatIP(embedding_numpy.shape[1])
# index.add(embedding_numpy)
# # Search for the nearest neighbors
# D, I = index.search(embedding_numpy, k)

#
# Tokenize the sentences
tokenized_sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
dataset = MyDataset(tokenized_sentences)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
#
#
#
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

    # Now we begin the discrete optimization process
    # We first get all the tokens that are not special tokens in the batch
    optimizer = BeamSearchOptimizer(model, batch_hidden_states, batch, device=device,
                                    neighbours=nearest_neighbors, tokenizer=tokenizer, beam_size=beam_size,
                                    iteration=10,
                                    alpha=0.7, beta=0.3)
    optimizer.initialize()
    dummy_data, loss = optimizer.optimize()
    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    print(tokenizer.decode(dummy_data[0], skip_special_tokens=True))
    transformed_data.append(tokenizer.decode(dummy_data[0], skip_special_tokens=True))
    i += 1
#
with open(f"{num_of_samples}_transformed_{data}_beam_{beam_size}_data_normalize.json", "w") as f:
    json.dump({"transformed_sentences": transformed_data, "labels": labels, "original_sentences": sentences}, f)