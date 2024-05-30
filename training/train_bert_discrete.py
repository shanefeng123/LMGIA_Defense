from sklearn.metrics import f1_score
from transformers import BertForSequenceClassification, BertTokenizerFast
from utils import *
import json
import torch
import random
from tqdm import tqdm
from argparse import ArgumentParser

# Set seed
random.seed(42)
torch.manual_seed(42)

parser = ArgumentParser("Transform data using discrete optimization")
parser.add_argument("--data", type=str, default="rotten_tomatoes", help="The dataset to use")
parser.add_argument("--beam_size", type=int, default=1, help="The beam size to use")
parser.add_argument("--batch_size", type=int, default=8, help="The batch size to use")
parser.add_argument("--num_of_samples", type=int, default=1000, help="The number of samples to use")
parser.add_argument("--model_name", type=str, default="bert-large-cased", help="The model to use")
parser.add_argument("--device", type=str, default="cuda:0", help="The device to use")
args = parser.parse_args()

data = args.data
beam_size = args.beam_size
batch_size = args.batch_size
num_of_samples = args.num_of_samples
model_name = args.model_name
device = args.device
dataset = json.load(open(f"../data/{model_name}/{num_of_samples}_transformed_{data}_beam_{beam_size}_data.json", "r"))
sentences = dataset["original_sentences"]
labels = dataset["labels"]
transformed_sentences = dataset["transformed_sentences"]

shuffled_indices = list(range(len(sentences)))
random.shuffle(shuffled_indices)
transformed_sentences = [transformed_sentences[i] for i in shuffled_indices]
sentences = [sentences[i] for i in shuffled_indices]
labels = [labels[i] for i in shuffled_indices]

labels = torch.LongTensor(labels)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model.to(device)

train_sentences = transformed_sentences[:int(num_of_samples * 0.8)]
train_labels = labels[:int(num_of_samples * 0.8)]

test_sentences = sentences[int(num_of_samples * 0.8):]
test_labels = labels[int(num_of_samples * 0.8):]

tokenized_train_set = tokenizer(train_sentences, return_tensors="pt", padding=True, truncation=True)
tokenized_train_set["labels"] = train_labels
train_dataset = MyDataset(tokenized_train_set)

tokenized_test_set = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True)
tokenized_test_set["labels"] = test_labels
test_dataset = MyDataset(tokenized_test_set)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
result_file = f"../results/{model_name}/{num_of_samples}_{data}_beam_{beam_size}_b_{batch_size}_result.txt"
best_test_f1 = 0
patience = 5
current_patience = 0
i = 0
while True:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loop = tqdm(train_loader, leave=True)
    overall_train_loss = 0
    train_predictions = []
    train_labels = []
    model.train()
    for training_batch in train_loop:
        optimizer.zero_grad()
        input_ids = training_batch["input_ids"].to(device)
        attention_mask = training_batch["attention_mask"].to(device)
        labels = training_batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        train_predictions.extend(predictions.tolist())
        train_labels.extend(labels.tolist())
        batch_loss = outputs.loss
        batch_loss.backward()
        optimizer.step()
        overall_train_loss += batch_loss.item()
    print(f"Epoch {i + 1} training: ")
    average_train_loss = overall_train_loss / len(train_loader)
    print("averaged_train_loss:", average_train_loss)
    train_accuracy = (torch.tensor(train_predictions) == torch.tensor(train_labels)).sum().item() / len(train_labels)
    print("train accuracy:", train_accuracy)
    train_f1 = f1_score(train_labels, train_predictions)
    print("train f1:", train_f1)
    with open(result_file, "a") as file:
        file.write(f"Epoch {i + 1} training: \n")
        file.write("averaged_train_loss: " + str(average_train_loss) + "\n")
        file.write("train accuracy: " + str(train_accuracy) + "\n")
        file.write("train f1: " + str(train_f1) + "\n")
        file.write("\n")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_loop = tqdm(test_loader, leave=True)
    overall_test_loss = 0
    test_predictions = []
    test_labels = []
    model.eval()
    for testing_batch in test_loop:
        input_ids = testing_batch["input_ids"].to(device)
        attention_mask = testing_batch["attention_mask"].to(device)
        labels = testing_batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        test_predictions.extend(predictions.tolist())
        test_labels.extend(labels.tolist())
        batch_loss = outputs.loss
        overall_test_loss += batch_loss.item()
    print(f"Epoch {i + 1} testing: ")
    average_test_loss = overall_test_loss / len(test_loader)
    print("averaged_test_loss:", average_test_loss)
    test_accuracy = (torch.tensor(test_predictions) == torch.tensor(test_labels)).sum().item() / len(test_labels)
    print("test accuracy:", test_accuracy)
    test_f1 = f1_score(test_labels, test_predictions)
    print("test f1:", test_f1)
    with open(result_file, "a") as file:
        file.write(f"Epoch {i + 1} testing: \n")
        file.write("averaged_test_loss: " + str(average_test_loss) + "\n")
        file.write("test accuracy: " + str(test_accuracy) + "\n")
        file.write("test f1: " + str(test_f1) + "\n")
        file.write("\n")

    if test_f1 > best_test_f1:
        best_test_f1 = test_f1
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
# model.save_pretrained("cola_model")
