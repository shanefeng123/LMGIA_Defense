from transformers import BertModel, BertTokenizerFast
from datasets import load_dataset
from utils import *
import json
import random

random.seed(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)
model.eval()
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
dataset = load_dataset("glue", "sst2")["train"]
sentences = dataset["sentence"]
labels = dataset["label"]

# Shuffle the sentences and labels together
shuffled_indices = list(range(len(sentences)))
random.shuffle(shuffled_indices)
sentences = [sentences[i] for i in shuffled_indices]
labels = [labels[i] for i in shuffled_indices]

# Get the first 1000 sentences
sentences = sentences[:1000]
labels = labels[:1000]

embedding_layer = model.embeddings.word_embeddings
transformed_embeddings = []

# Tokenize the sentences
tokenized_sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
# Record the index of the [SEP] token for each sentence
sep_indices = []
for i in range(len(tokenized_sentences["input_ids"])):
    sep_indices.append((tokenized_sentences["input_ids"][i] == tokenizer.sep_token_id).nonzero().item())
tokenized_sentences["sep_indices"] = sep_indices

dataset = MyDataset(tokenized_sentences)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

for i in range(len(dataloader)):
    batch = next(iter(dataloader))
    print("Processing batch ", i + 1, "out of", len(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    batch_embeddings = embedding_layer(input_ids)
    batch_outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    batch_hidden_states = batch_outputs.hidden_states

    # Initialize dummy data with the same value as batch embeddings, but track the gradients
    dummy_embeddings = batch_embeddings.clone().detach()
    # Add some noise to the dummy embeddings
    dummy_embeddings += torch.randn_like(dummy_embeddings) * 0.1
    dummy_embeddings = dummy_embeddings.requires_grad_(True)
    # Replace the embeddings of [CLS], [SEP], and [PAD] tokens with the original embeddings
    with torch.no_grad():
        for j in range(len(dummy_embeddings)):
            dummy_embeddings[j][0] = batch_embeddings[j][0]
            dummy_embeddings[j][batch["sep_indices"][j]] = batch_embeddings[j][batch["sep_indices"][j]]
            dummy_embeddings[j][batch["attention_mask"][j] == 0] = batch_embeddings[j][batch["attention_mask"][j] == 0]

    # Optimize the dummy_sample with gradient descent, such that the dummy_sample has similar output logits but different embeddings
    optimizer = torch.optim.Adam([dummy_embeddings], lr=0.01)
    for _ in range(1000):
        optimizer.zero_grad()
        dummy_outputs = model(inputs_embeds=dummy_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        dummy_hidden_states = dummy_outputs.hidden_states
        # output similarity loss is the MSE between the hidden states
        output_similarity_loss = 0
        for j in range(len(batch_hidden_states)):
            output_similarity_loss += torch.nn.functional.mse_loss(batch_hidden_states[j], dummy_hidden_states[j])
        # embedding dissimilarity loss is the MSE between the embeddings excluding [CLS], [SEP], and [PAD] tokens
        embedding_dissimilarity_loss = torch.nn.functional.mse_loss(dummy_embeddings[batch["attention_mask"] == 1],
                                                                    batch_embeddings[batch["attention_mask"] == 1])
        # embedding_dissimilarity_loss = torch.nn.functional.mse_loss(dummy_embeddings, batch_embeddings)
        loss = output_similarity_loss - embedding_dissimilarity_loss
        loss.backward(retain_graph=True)
        # print(loss.item())
        optimizer.step()
        # 　Do not update the embeddings of [CLS], [SEP], and [PAD] tokens
        with torch.no_grad():
            for j in range(len(dummy_embeddings)):
                dummy_embeddings[j][0] = batch_embeddings[j][0]
                dummy_embeddings[j][batch["sep_indices"][j]] = batch_embeddings[j][batch["sep_indices"][j]]
                dummy_embeddings[j][batch["attention_mask"][j] == 0] = batch_embeddings[j][batch["attention_mask"][j] == 0]

    # Record the transformed embeddings one by one in the batch
    transformed_embeddings.extend(dummy_embeddings.detach().cpu().tolist())
    # break

with open("sst2_data.json", "w") as f:
    json.dump({"sentences": sentences, "labels": labels, "transformed_embeddings": transformed_embeddings}, f)
