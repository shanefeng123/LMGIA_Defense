from transformers import BertForSequenceClassification, BertTokenizerFast
from utils import *
import json
import torch
from tqdm import tqdm

device = "cuda:0"
data = json.load(open("cola_data.json", "r"))
sentences = data["sentences"]
labels = data["labels"]
transformed_embeddings = data["transformed_embeddings"]
transformed_embeddings = torch.tensor(transformed_embeddings)
labels = torch.LongTensor(labels)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model.to(device)
# Tokenize the sentences
tokenized_sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
# Get the first 800 of the transformed embeddings as training data
train_transformed_embeddings = transformed_embeddings[:800]
train_labels = labels[:800]
train_attention_mask = tokenized_sentences["attention_mask"][:800]
dummy_train_set = {"train_transformed_embeddings": train_transformed_embeddings,
                   "train_attention_mask": train_attention_mask, "train_labels": train_labels}
dummy_train_set = MyDummyDataset(dummy_train_set)
# Get the last 200 of the original sentences as test data
test_input_ids = tokenized_sentences["input_ids"][800:]
test_attention_mask = tokenized_sentences["attention_mask"][800:]
test_labels = labels[800:]
test_dataset = MyDataset({"input_ids": test_input_ids, "attention_mask": test_attention_mask, "labels": test_labels})

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
result_file = "cola_result.txt"
best_test_accuracy = 0
patience = 5
current_patience = 0
i = 0
while True:
    train_loader = torch.utils.data.DataLoader(dummy_train_set, batch_size=8, shuffle=True)
    train_loop = tqdm(train_loader, leave=True)
    overall_train_loss = 0
    train_predictions = []
    train_labels = []
    for training_batch in train_loop:
        model.train()
        optimizer.zero_grad()
        transformed_embeddings = training_batch["train_transformed_embeddings"].to(device)
        attention_mask = training_batch["train_attention_mask"].to(device)
        labels = training_batch["train_labels"].to(device)
        outputs = model(inputs_embeds=transformed_embeddings, attention_mask=attention_mask, labels=labels)
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
    with open(result_file, "a") as file:
        file.write(f"Epoch {i + 1} training: \n")
        file.write("averaged_train_loss: " + str(average_train_loss) + "\n")
        file.write("train accuracy: " + str(train_accuracy) + "\n")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)
    test_loop = tqdm(test_loader, leave=True)
    overall_test_loss = 0
    test_predictions = []
    test_labels = []
    for testing_batch in test_loop:
        model.eval()
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
    with open(result_file, "a") as file:
        file.write(f"Epoch {i + 1} testing: \n")
        file.write("averaged_test_loss: " + str(average_test_loss) + "\n")
        file.write("test accuracy: " + str(test_accuracy) + "\n")
        file.write("\n")

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        current_patience = 0
    else:
        current_patience += 1
        if current_patience == patience:
            print("Early stopping")
            with open(result_file, "a") as file:
                file.write(f"Early stopping at epoch {i + 1}\n")
                file.write(f"Best model is at epoch {i}")
            break
    i += 1
# model.save_pretrained("cola_model")



