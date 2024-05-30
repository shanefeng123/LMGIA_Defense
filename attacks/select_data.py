import json
import torch
import random
from argparse import ArgumentParser

# Set seed
random.seed(42)
torch.manual_seed(42)

parser = ArgumentParser("Selecting data to attack on")
parser.add_argument("--data", type=str, default="rotten_tomatoes", help="The dataset to use")
parser.add_argument("--beam_size", type=int, default=1, help="The beam size to use")
parser.add_argument("--select_size", type=int, default=64, help="The select size to use")
parser.add_argument("--num_of_samples", type=int, default=1000, help="The number of samples to use")
args = parser.parse_args()

device = "cuda:0"
data = args.data
beam_size = args.beam_size
select_size = args.select_size
num_of_samples = args.num_of_samples
dataset = json.load(open(f"../data/mlm/{num_of_samples}_transformed_{data}_beam_{beam_size}_data.json", "r"))
sentences = dataset["original_sentences"]
labels = dataset["labels"]
transformed_sentences = dataset["transformed_sentences"]

shuffled_indices = list(range(len(sentences)))
random.shuffle(shuffled_indices)
selected_indices = shuffled_indices[:select_size]
transformed_sentences = [transformed_sentences[i] for i in selected_indices]
sentences = [sentences[i] for i in selected_indices]
labels = [labels[i] for i in selected_indices]

# Save the selected data
selected_data = {"original_sentences": sentences, "transformed_sentences": transformed_sentences, "labels": labels}
with open(f"data/{data}_selected_data.json", "w") as file:
    json.dump(selected_data, file)
