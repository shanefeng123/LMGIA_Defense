import copy
import math
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from utils import *
import json
import torch
import random
from tqdm import tqdm
from argparse import ArgumentParser

# Set seed
random.seed(42)
torch.manual_seed(42)

parser = ArgumentParser("Train GPT2 model with transformed data")
parser.add_argument("--data", type=str, default="enron", help="The dataset to use")
parser.add_argument("--beam_size", type=int, default=1, help="The beam size to use")
parser.add_argument("--batch_size", type=int, default=8, help="The batch size to use")
parser.add_argument("--num_of_samples", type=int, default=1000, help="The number of samples to use")
parser.add_argument("--model_name", type=str, default="gpt2-xl", help="The model to use")
parser.add_argument("--device", type=str, default="cuda:0", help="The device to use")
args = parser.parse_args()

device = args.device
data = args.data
beam_size = args.beam_size
batch_size = args.batch_size
model_name = args.model_name
num_of_samples = args.num_of_samples

dataset = json.load(open(f"../data/{model_name}/{num_of_samples}_transformed_{data}_beam_{beam_size}_data.json", "r"))
sentences = dataset["original_sentences"]
transformed_sentences = dataset["transformed_sentences"]

for i in range(len(sentences)):
    transformed_sentences[i] = "<|startoftext|>" + transformed_sentences[i] + "<|endoftext|>"

shuffled_indices = list(range(len(sentences)))
random.shuffle(shuffled_indices)
transformed_sentences = [transformed_sentences[i] for i in shuffled_indices]
sentences = [sentences[i] for i in shuffled_indices]

tokenizer = GPT2TokenizerFast.from_pretrained(model_name, bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.resize_token_embeddings(len(tokenizer))
model.eval()

train_sentences = transformed_sentences[:int(num_of_samples * 0.8)]
test_sentences = sentences[int(num_of_samples * 0.8):]

tokenized_train_set = tokenizer(train_sentences, return_tensors="pt", padding=True, truncation=True)
tokenized_train_set["labels"] = tokenized_train_set["input_ids"].clone()
train_dataset = MyDataset(tokenized_train_set)

tokenized_test_set = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True)
tokenized_test_set["labels"] = tokenized_test_set["input_ids"].clone()
test_dataset = MyDataset(tokenized_test_set)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
result_file = f"../results/{model_name}/{num_of_samples}_{data}_beam_{beam_size}_b_{batch_size}.txt"
best_test_perplexity = math.inf
best_model = copy.deepcopy(model)
patience = 5
current_patience = 0
i = 0
while True:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loop = tqdm(train_loader, leave=True)
    overall_train_loss = 0
    model.train()
    for training_batch in train_loop:
        optimizer.zero_grad()
        input_ids = training_batch["input_ids"].to(device)
        attention_mask = training_batch["attention_mask"].to(device)
        labels = training_batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        batch_loss = outputs.loss
        batch_loss.backward()
        optimizer.step()
        overall_train_loss += batch_loss.item()
    print(f"Epoch {i + 1} training: ")
    average_train_loss = overall_train_loss / len(train_loader)
    print("averaged_train_loss:", average_train_loss)
    average_train_perplexity = math.exp(average_train_loss)
    print("average_train_perplexity:", average_train_perplexity)
    with open(result_file, "a") as file:
        file.write(f"Epoch {i + 1} training: \n")
        file.write("averaged_train_loss: " + str(average_train_loss) + "\n")
        file.write("average_train_perplexity: " + str(average_train_perplexity) + "\n")
        file.write("\n")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loop = tqdm(test_loader, leave=True)
    overall_test_loss = 0
    model.eval()
    for testing_batch in test_loop:
        input_ids = testing_batch["input_ids"].to(device)
        attention_mask = testing_batch["attention_mask"].to(device)
        labels = testing_batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        batch_loss = outputs.loss
        overall_test_loss += batch_loss.item()
    print(f"Epoch {i + 1} testing: ")
    average_test_loss = overall_test_loss / len(test_loader)
    print("averaged_test_loss:", average_test_loss)
    average_test_perplexity = math.exp(average_test_loss)
    print("average_test_perplexity:", average_test_perplexity)
    with open(result_file, "a") as file:
        file.write(f"Epoch {i + 1} testing: \n")
        file.write("averaged_test_loss: " + str(average_test_loss) + "\n")
        file.write("average_test_perplexity: " + str(average_test_perplexity) + "\n")
        file.write("\n")

    if average_test_perplexity < best_test_perplexity:
        best_test_perplexity = average_test_perplexity
        best_model = copy.deepcopy(model)
        current_patience = 0
    else:
        current_patience += 1
        if current_patience == patience:
            print("Early stopping")

            print(f"Best model is at epoch {i + 1 - patience}")
            with open(result_file, "a") as file:
                file.write(f"Early stopping at epoch {i + 1}\n")
                file.write(f"Best model is at epoch {i + 1 - patience}")
            break
    i += 1

best_model.eval()
with torch.no_grad():
    generated = best_model.generate(
        max_length=200,
        num_return_sequences=5,
        # temperature=1.2,
        top_k=50,
        # top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id)
    for g in generated:
        print(tokenizer.decode(g, skip_special_tokens=True).strip())
