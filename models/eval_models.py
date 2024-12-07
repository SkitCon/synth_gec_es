#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("..")

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

from models.model import BETOTokenLevelGECModel, Seq2SeqTextDataset, TokenLevelGECDataset
from utils.utils import load_modified_nlp, load_vocab, load_morpho_dict, labels_to_vec

import os
os.chdir("..")

from torch.utils.data import Dataset, DataLoader


# In[ ]:


input_data = "data/COWS-L2H-labeled-STRICT.txt"

max_len = 128
max_labels = 12
batch_size = 16


# In[ ]:


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
print(f"Ignored {ignored_sentences} sentences.")

train, test = train_test_split(samples, train_size=0.8, random_state=42)

# Only take 2000 because of time constraints
test = test[:2000]


# In[1]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_model(model, tokenizer, dataloader, is_seq2seq=True):
    model.eval()
    predictions, references = [], []

    errorful = []

    if not is_seq2seq:
        lemma_to_morph = load_morpho_dict("lang_def/morpho_dict_updated.json")
        vocab = load_vocab("lang_def/vocab.txt")
        nlp = load_modified_nlp()
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if is_seq2seq:
            correct_sentences = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        else:
            correct_sentences = batch["correct_sentence"]
        
        errorful.extend(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
        references.extend([[ref.split()] for ref in correct_sentences])
        
        if is_seq2seq:
            # Generate predictions
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64)

            # Decode predictions and labels
            predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        else:
            type_logits, param_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            sentences = [tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids]
            predictions.extend(model.batch_decode_from_logits(sentences, type_logits, param_logits, lemma_to_morph, vocab, nlp))
    
    errorful = [sentence.split() for sentence in errorful]
    predictions = [pred.split() for pred in predictions]
    bleu_score = corpus_bleu(references, predictions)
    print(f"Base BLEU Score: {bleu_score:.4f}")

    no_error = [(predictions[i], references[i], errorful[i]) for i in range(len(predictions)) if errorful[i] == references[i][0]]
    has_error = [(predictions[i], references[i], errorful[i]) for i in range(len(predictions)) if errorful[i] != references[i][0]]

    count_tp = len([i for i in range(len(has_error)) if has_error[i][0] == has_error[i][1][0]])
    count_fp = len([i for i in range(len(no_error)) if no_error[i][0] != no_error[i][1][0]])
    count_tn = len([i for i in range(len(no_error)) if no_error[i][0] == no_error[i][1][0]])
    count_fn = len([i for i in range(len(has_error)) if has_error[i][0] == has_error[i][2]])
    incomplete_corrections = len(predictions) - (count_tp + count_fp + count_tn + count_fn)

    print(f"Based on detection of error:\nTP: {count_tp}\nFP: {count_fp}\nTN: {count_tn}\nFN: {count_fn}\nError was detected, but correction was incomplete: {incomplete_corrections}")

    if len(no_error) > 0:
        no_error_preds, no_error_refs, no_error_errorful = zip(*no_error)
        print(f"BLEU score on sentences with no errors: {corpus_bleu(no_error_refs, no_error_preds)}")
    else:
        print(f"BLEU score on sentences with no errors: NO BLEU BECAUSE NO SENTENCES WITH NO ERROR")

    if len(has_error) > 0:
        error_preds, error_refs, error_errorful = zip(*has_error)
        print(f"BLEU score on sentences with errors: {corpus_bleu(error_refs, error_preds)}")
    else:
        print(f"BLEU score on sentences with errors: NO BLEU BECAUSE NO SENTENCES WITH ERRORS")
        

def test_selected_sentences(model, tokenizer, is_seq2seq=True):
    sentences = [("yo va al tienda.", ["Yo voy a la tienda.", "Voy a la tienda.", "Va a la tienda."]), \
                 ("Gracias para invitarme.", ["Gracias por invitarme."]), \
                 ("Espero que tú ganas el juego.", ["Espero que ganes el juego.", "Espero que tú ganes el juego."]), \
                 ("Soy una mujer y soy bello.", ["Soy una mujer y soy bella."]), \
                 ("Le dije que estaba avergonzada, pero me llamó emotivo.", ["Le dije que estaba avergonzada, pero me llamó emotiva."]), \
                 ("Mi novia me dijo que me parecía bella esta noche.", ["Mi novia me dijo que me parecía bella esta noche."]), \
                 ("Trabajo como albañil así que siempre me siento cansada.", ["Trabajo como albañil así que siempre me siento cansada."])]

    if not is_seq2seq:
        lemma_to_morph = load_morpho_dict("lang_def/morpho_dict_updated.json")
        vocab = load_vocab("lang_def/vocab.txt")
        nlp = load_modified_nlp()
    
    print("-----------------------\nRunning selected tests...")
    
    errorful, correct = zip(*sentences)
    tokenized_text = tokenizer(errorful, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

    input_ids = tokenized_text["input_ids"].squeeze().to(device)
    attention_mask = tokenized_text["attention_mask"].squeeze().to(device)

    if is_seq2seq:
        # Generate predictions
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64)

        # Decode predictions and labels
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    else:
        type_logits, param_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        output_sentences = [tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids]
        predictions = model.batch_decode_from_logits(output_sentences, type_logits, param_logits, lemma_to_morph, vocab, nlp)
    
    for i in range(len(predictions)):
        if not predictions[i] in correct[i]:
            print(f"FAILED TEST {i+1}: Target: {correct[i]}, Result: {predictions[i]}")


# In[ ]:


seq2seq_base_tokenizer = AutoTokenizer.from_pretrained("SkitCon/gec-spanish-BARTO-COWS-L2H")
seq2seq_base_model = BartForConditionalGeneration.from_pretrained("SkitCon/gec-spanish-BARTO-COWS-L2H").to(device)

errorful, labels, correct = zip(*test)

test_dataset = Seq2SeqTextDataset(errorful, correct, seq2seq_base_tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

print("=======================\nEval on seq2seq base:")
eval_model(seq2seq_base_model, seq2seq_base_tokenizer, test_dataloader, is_seq2seq=True)
test_selected_sentences(seq2seq_base_model, seq2seq_base_tokenizer, is_seq2seq=True)

seq2seq_synth_tokenizer = AutoTokenizer.from_pretrained("SkitCon/gec-spanish-BARTO-SYNTHETIC")
seq2seq_synth_model = BartForConditionalGeneration.from_pretrained("SkitCon/gec-spanish-BARTO-SYNTHETIC").to(device)

print("=======================\nEval on seq2seq synth:")
eval_model(seq2seq_synth_model, seq2seq_synth_tokenizer, test_dataloader, is_seq2seq=True)
test_selected_sentences(seq2seq_synth_model, seq2seq_synth_tokenizer, is_seq2seq=True)

token_base_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
token_base_model = BETOTokenLevelGECModel.from_pretrained("SkitCon/gec-spanish-BETO-TOKEN-COWS-L2H").to(device)

test_dataset = TokenLevelGECDataset(*zip(*test), token_base_tokenizer, max_len=max_len)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

print("=======================\nEval on token base:")
eval_model(token_base_model, token_base_tokenizer, test_dataloader, is_seq2seq=False)
test_selected_sentences(token_base_model, token_base_tokenizer, is_seq2seq=False)

token_synth_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
token_synth_model = BETOTokenLevelGECModel.from_pretrained("SkitCon/gec-spanish-BETO-TOKEN-SYNTHETIC").to(device)

print("=======================\nEval on token synth:")
eval_model(token_synth_model, token_synth_tokenizer, test_dataloader, is_seq2seq=False)
test_selected_sentences(token_synth_model, token_synth_tokenizer, is_seq2seq=False)

