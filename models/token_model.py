#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append("..")

import warnings
warnings.filterwarnings("ignore")

import time
import argparse
import spacy
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertConfig, BertModel, PreTrainedModel, PretrainedConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

from utils.utils import load_morpho_dict, load_vocab, load_modified_nlp, labels_to_vec, vec_to_labels, apply_labels

import os
os.chdir("..")

torch.manual_seed(42)


# In[146]:


# Experiment variables

parser = argparse.ArgumentParser()
parser.add_argument("--synth",
                    help="Use synthetic data",
                    action="store_true")
args = parser.parse_args()

input_data = "data/COWS-L2H-labeled-STRICT.txt"
if args.synth:
    synthetic_data = "data/batch_1_synthetic.txt"
    output_path = "./token_model_SYNTHETIC"
else:
    synthetic_data = ""
    output_path = "./token_model"
    
model_name = "dccuchile/bert-base-spanish-wwm-cased"

max_labels = 12
max_len = 128

epochs = 8
batch_size = 16

train_size = .8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[147]:


class TokenLevelGECDataset(Dataset):
    def __init__(self, errorful_sentences, labels, correct_sentences, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        encoded = [tokenizer(errorful_sentence,
                             padding="max_length",
                             truncation=True,
                             max_length=max_len,
                             return_tensors="pt") for errorful_sentence in errorful_sentences]
        seq_lens = [len(encoding) for encoding in encoded]
        self.samples = list(zip(encoded, labels, correct_sentences, seq_lens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded, labels, correct, seq_len = self.samples[idx]
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
            "correct_sentence": correct,
            "seq_len": seq_len
        }


# In[149]:


class BETOTokenLevelGECModel(BertModel):
    config_class = BertConfig
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.encoder = BertModel(config)
        hidden_size = self.encoder.config.hidden_size
        self.max_labels = kwargs.get("max_labels", 12)
        self.vocab_size = config.vocab_size
        
        self.type_classifier = nn.Linear(hidden_size, 7 * self.max_labels)
        self.param_classifier = nn.Linear(hidden_size, self.vocab_size * self.max_labels)
        
        self.post_init()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state[:, 1:, :]  # Shape: (batch_size, max_len-1, hidden_size), exclude [CLS]
        
        # Predict transformation types
        type_logits = self.type_classifier(token_embeddings)
        type_logits = type_logits.view(token_embeddings.size(0), token_embeddings.size(1), self.max_labels, 7)

        # Predict transformation parameters
        param_logits = self.param_classifier(token_embeddings)  # Shape: (B, L, max_labels * vocab_size)
        param_logits = param_logits.view(token_embeddings.size(0), token_embeddings.size(1), self.max_labels, self.vocab_size)

        return type_logits, param_logits

    def logits_to_vec(self, type_logits, param_logits):
        # Apply softmax to get probabilities
        type_probs = torch.softmax(type_logits, dim=-1)  # Shape: (batch_size, max_len, max_labels, 7)
        param_probs = torch.softmax(param_logits, dim=-1)  # Shape: (batch_size, max_len, max_labels, vocab_size)
        
        # Select the most probable type and parameter for each transformation pair
        pred_types = torch.argmax(type_probs, dim=-1) # Shape: (batch_size, max_len, max_labels)
        pred_params = torch.argmax(param_probs, dim=-1) # Shape: (batch_size, max_len, max_labels)
    
        # Ok I don't mean to brag but this was so big brain... took me way too long to figure this out...
        assert pred_types.shape == pred_params.shape
        pred_vec = torch.stack((pred_types, pred_params), dim=3).view(-1, pred_types.shape[1], 2*pred_types.shape[2])
        
        return pred_vec

    def decode(self, sentence, vec, lemma_to_morph, vocab, nlp):
        sentence_doc = nlp(sentence)
        labels = vec_to_labels(vec, seq_len=len(sentence_doc))
        try:
            return apply_labels(sentence_doc, labels.split('\t'), lemma_to_morph, vocab, nlp)
        except KeyError as e:
            print(f"Failed to decode sentence due to KeyError: {e} (invalid mutation?)")
            return sentence
        except IndexError as e:
            print(f"Failed to decode sentence due to IndexError: {e} (invalid copy?)")
            return sentence

    def batch_decode_from_logits(self, sentences, type_logits, param_logits, lemma_to_morph, vocab, nlp):
        vecs = self.logits_to_vec(type_logits, param_logits).cpu()
        decoded_sentences = []
        for i, sentence in enumerate(sentences):
            decoded_sentences.append(self.decode(sentence, vecs[i], lemma_to_morph, vocab, nlp))
        return decoded_sentences


# In[150]:


def evaluate_model(model, dataloader, tokenizer, device, max_labels, max_len=128):

    lemma_to_morph = load_morpho_dict("lang_def/morpho_dict_updated.json")
    vocab = load_vocab("lang_def/vocab.txt")
    nlp = load_modified_nlp()
    
    model.eval()
    predictions, references = [], []

    start_time = time.time()
    num_complete = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            correct_sentences = batch["correct_sentence"]
            seq_lens = batch["seq_len"]

            type_logits, param_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            sentences = [tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids]
            predictions.extend(model.batch_decode_from_logits(sentences, type_logits, param_logits, lemma_to_morph, vocab, nlp))
            references.extend(correct_sentences)

            num_complete += len(batch)

            if i in list(range(100, len(dataloader), 100)):
                cur_time = time.time()
                print(f"Evaluated {num_complete} batches in {(cur_time - start_time) / 60:.2f} minutes.")

    # Evaluate BLEU
    from nltk.translate.bleu_score import corpus_bleu
    bleu_score = corpus_bleu([[ref.split()] for ref in references], [pred.split() for pred in predictions])
    print(f"BLEU Score: {bleu_score:.4f}")
    return bleu_score


# In[ ]:


# Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BETOTokenLevelGECModel.from_pretrained(model_name, max_labels=max_labels).to(device)

# I'm not sure how I'm supposed to do this, but for some reason the encoder is not initialized with the weights from BETO.
# So I overwrite it the FIRST time it is trained, then it loads fine from the checkpoint.
print("IMPORTANT: Ignore previous weights warning. I am overwriting the encoder that throws the previous initialization warning.")
model.encoder = AutoModel.from_pretrained(model_name).to(device)


# In[153]:


# Load data

samples = []
ignored_sentences = 0
with open(input_data, 'r') as f:
    lines = f.readlines()
    for i in range(0, len(lines), 4):
        errorful = lines[i].strip()
        transformations = lines[i + 1].strip()
        correct = lines[i + 2].strip()

        if max([len(token_labels.split(' ')) for token_labels in transformations.split('\t')]) > max_labels:
            # print("Discarding sentence, too many labels, likely sentence mismatch")
            samples.append(("", labels_to_vec("", max_len=max_len, max_labels=max_labels), ""))
            ignored_sentences += 1
        elif len(transformations.split('\t')) > max_len:
            # print("Sentence too long, discarding sentence for training")
            samples.append(("", labels_to_vec("", max_len=max_len, max_labels=max_labels), ""))
            ignored_sentences += 1
        else:
            try:
                label_vecs = labels_to_vec(transformations, max_len=max_len, max_labels=max_labels)
            except KeyError as e:
                # print(f"{e} not a valid label. Discarding sentence.")
                label_vecs = labels_to_vec("", max_len=max_len, max_labels=max_labels)
                ignored_sentences += 1
            samples.append((errorful, label_vecs, correct))
train, test = train_test_split(samples, train_size=0.8, random_state=42)

# Only take 2000 because of time constraints
test = test[:2000]

if synthetic_data:
    synthetic_samples = []
    with open(synthetic_data, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):
            errorful = lines[i].strip()
            transformations = lines[i + 1].strip()
            correct = lines[i + 2].strip()
    
            if max([len(token_labels.split(' ')) for token_labels in transformations.split('\t')]) > max_labels:
                # print("Discarding sentence, too many labels, likely sentence mismatch")
                synthetic_samples.append(("", labels_to_vec("", max_len=max_len, max_labels=max_labels), ""))
                ignored_sentences += 1
            elif len(transformations.split('\t')) > max_len:
                # print("Sentence too long, discarding sentence for training")
                synthetic_samples.append(("", labels_to_vec("", max_len=max_len, max_labels=max_labels), ""))
                ignored_sentences += 1
            else:
                try:
                    label_vecs = labels_to_vec(transformations, max_len=max_len, max_labels=max_labels)
                except KeyError as e:
                    # print(f"{e} not a valid label. Discarding sentence.")
                    label_vecs = labels_to_vec("", max_len=max_len, max_labels=max_labels)
                    ignored_sentences += 1
                synthetic_samples.append((errorful, label_vecs, correct))
    train += synthetic_samples
print(f"Ignored {ignored_sentences} sentences.")

train_dataset = TokenLevelGECDataset(*zip(*train), tokenizer, max_len=max_len)
test_dataset = TokenLevelGECDataset(*zip(*test), tokenizer, max_len=max_len)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# In[154]:


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        type_logits, param_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        type_labels = labels[:, :-1, 0::2].contiguous()
        param_labels = labels[:, :-1, 1::2].contiguous()
        
        # Flatten the dimensions for loss calculation
        type_loss = torch.nn.functional.cross_entropy(type_logits.view(-1, 7), type_labels.view(-1))
        param_loss = torch.nn.functional.cross_entropy(param_logits.view(-1, model.vocab_size), param_labels.view(-1))
        
        # Combine losses (weighted equally)
        loss = type_loss + param_loss
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader):.4f}")


# In[1]:


tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path)
evaluate_model(model, test_dataloader, tokenizer, device, max_labels, max_len=128)

