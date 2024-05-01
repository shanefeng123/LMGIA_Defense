from transformers import BertModel, BertTokenizerFast
from utils import *
from collections import Counter
import json
import torch
from tqdm import tqdm

device = "cuda:0"
data = json.load(open("1000_transformed_sst2_beam_1_data.json", "r"))
sentences = data["original_sentences"]
labels = data["labels"]
transformed_sentences = data["transformed_sentences"]
labels = torch.LongTensor(labels)
model = BertModel.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_sentence = transformed_sentences[0]
ori_sentence = sentences[0]

ori_pooler_output = model(**tokenizer(ori_sentence, return_tensors="pt")).pooler_output
train_pooler_output = model(**tokenizer(train_sentence, return_tensors="pt")).pooler_output
# cosine similarity
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
print(cos(ori_pooler_output, train_pooler_output))

# test_sentences = sentences[800:]
# test_labels = labels[800:]
# print(Counter(test_labels))