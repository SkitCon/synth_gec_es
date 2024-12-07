#!/usr/bin/env python
# coding: utf-8

# In[10]:


import time
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

import os
os.chdir("..")

torch.manual_seed(42)


# In[11]:


# Experiment variables

parser = argparse.ArgumentParser()
parser.add_argument("--synth",
                    help="Use synthetic data",
                    action="store_true")
args = parser.parse_args()

input_data = "data/COWS-L2H-unlabeled-STRICT.txt"
if args.synth:
    synthetic_data = "data/batch_1_synthetic.txt"
    synthetic_data_is_labeled = True
    output_path = "./seq2seq_gec_model_SYNTHETIC"
else:
    synthetic_data = ""
    output_path = "./seq2seq_gec_model"
model_name = "vgaraujov/bart-base-spanish"

epochs = 8
batch_size = 32

train_size = .8


# In[12]:


# Dataset Definition

class Seq2SeqTextDataset(Dataset):
    def __init__(self, x, y, tokenizer, max_length=128):
        self.samples = list(zip(x, y))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source, target = self.samples[idx]
        source_enc = self.tokenizer(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": target_enc["input_ids"].squeeze(),
        }


# In[13]:


def load_data(file_path, is_labeled=False, use_labels=False):
    samples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if is_labeled:
        for i in range(0, len(lines), 4):  # Assumes blank line separates pairs
            if i + 2 < len(lines):
                source = lines[i].strip()
                target = lines[i+2].strip()
                if source and target:
                    samples.append((source, target))
    else:
        for i in range(0, len(lines), 3):  # Assumes blank line separates pairs
            if i + 1 < len(lines):
                source = lines[i].strip()
                target = lines[i+1].strip()
                if source and target:
                    samples.append((source, target))
    return samples

def eval_model_bleu(model, dataloader, device="cuda"):
    model.eval()
    references = []  # List of reference sentences (ground truth)
    predictions = []  # List of generated sentences (model predictions)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Generate predictions
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64)

            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Append references and references
            references.extend([[ref.split()] for ref in decoded_labels])
            predictions.extend([pred.split() for pred in decoded_preds])

    # Calculate BLEU score
    return corpus_bleu(references, predictions)


# In[23]:


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)


# In[31]:


# Load dataset

base_data = load_data(input_data)
if synthetic_data:
    synth_x, synth_y = zip(*load_data(synthetic_data, is_labeled=synthetic_data_is_labeled))

train_x, test_x, train_y, test_y = train_test_split(*zip(*base_data), train_size=0.8, random_state=42)

if synthetic_data:
    train_x += synth_x
    train_y += synth_y

train_dataset = Seq2SeqTextDataset(train_x, train_y, tokenizer)
test_dataset = Seq2SeqTextDataset(test_x, test_y, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# In[ ]:


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

num_batches = len(train_dataloader)

# Training loop
for epoch in range(epochs):

    start_time = time.time()
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_time = time.time()
        if i in list(range(0, num_batches, 500)):
            print(f"=====================\nProgress Report:\nFinished {i+1}/{num_batches} batches.\nTime per batch: {(cur_time-start_time)/(i+1):.2f} seconds.\nEstimated time remaining in epoch: {((num_batches-(i+1))*((cur_time-start_time)/(i+1)))/60:.2f} minutes")
    train_loss = total_loss / len(train_dataloader)

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
    test_loss = total_loss / len(test_dataloader)
    
    if not synthetic_data:
        train_bleu_score = eval_model_bleu(model, train_dataloader, device=device)
    test_bleu_score = eval_model_bleu(model, test_dataloader, device=device)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {test_loss:.4f}")
    if not synthetic_data:
        print(f"Train BLEU: {train_bleu_score:.4f}")
    print(f"Test BLEU: {test_bleu_score:.4f}")

train_bleu_score = eval_model_bleu(model, train_dataloader, device=device)
print(f"Train BLEU: {train_bleu_score:.4f}")
# Save model
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_path)

