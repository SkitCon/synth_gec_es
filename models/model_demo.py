#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("..")

import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from models.model import BETOTokenLevelGECModel
from utils.utils import load_modified_nlp, load_vocab, load_morpho_dict, vec_to_labels

import os
os.chdir("..")

seq2seq_base_tokenizer = AutoTokenizer.from_pretrained("SkitCon/gec-spanish-BARTO-COWS-L2H")
seq2seq_base_model = BartForConditionalGeneration.from_pretrained("SkitCon/gec-spanish-BARTO-COWS-L2H")

seq2seq_synth_tokenizer = AutoTokenizer.from_pretrained("SkitCon/gec-spanish-BARTO-SYNTHETIC")
seq2seq_synth_model = BartForConditionalGeneration.from_pretrained("SkitCon/gec-spanish-BARTO-SYNTHETIC")

token_base_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
token_base_model = BETOTokenLevelGECModel.from_pretrained("SkitCon/gec-spanish-BETO-TOKEN-COWS-L2H")

token_synth_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
token_synth_model = BETOTokenLevelGECModel.from_pretrained("SkitCon/gec-spanish-BETO-TOKEN-SYNTHETIC")

lemma_to_morph = load_morpho_dict("lang_def/morpho_dict_updated.json")
vocab = load_vocab("lang_def/vocab.txt")
nlp = load_modified_nlp()

user_input = input("=========================\nWelcome to the very simple demo of the four GEC models.\nChoose a model:\n\tSeq2Seq (COWS-L2H only): 1\n\tSeq2Seq (COWS-L2H + Synthetic): 2\n\tToken-Classification (COWS-L2H only): 3\n\tToken-Classification (COWS-L2H + Synthetic): 4\nOr type exit to leave.\n")
while user_input.lower() != "exit":
    if user_input in ["1", "2"]:
        tokenizer = seq2seq_base_tokenizer if user_input == "1" else seq2seq_synth_tokenizer
        model = seq2seq_base_model if user_input == "1" else seq2seq_synth_model

        input_sentence = input("\nType a sentence to correct:\n")

        tokenized_text = tokenizer([input_sentence, ""], max_length=128, padding="max_length", truncation=True, return_tensors="pt")

        input_ids = tokenized_text["input_ids"].squeeze()
        attention_mask = tokenized_text["attention_mask"].squeeze()

        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)

        correct_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        print(f"\nCorrected: {correct_sentence}")
    elif user_input in ["3", "4"]:

        tokenizer = token_base_tokenizer if user_input == "3" else token_synth_tokenizer
        model = token_base_model if user_input == "3" else token_synth_model
        
        input_sentence = input("\nType a sentence to correct:\n")

        tokenized_text = tokenizer([input_sentence, ""], max_length=128, padding="max_length", truncation=True, return_tensors="pt")

        input_ids = tokenized_text["input_ids"].squeeze()
        attention_mask = tokenized_text["attention_mask"].squeeze()

        type_logits, param_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        print(f"\nLabels: {vec_to_labels(model.logits_to_vec(type_logits, param_logits)[0], seq_len=len(nlp(input_sentence)))}")

        sentences = [tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids]
        
        correct_sentence = model.batch_decode_from_logits(sentences, type_logits, param_logits, lemma_to_morph, vocab, nlp)[0]
        
        print(f"Corrected: {correct_sentence}")
    else:
        user_input = input("\n=========================\nPick a valid model (1, 2, 3, or 4)")
        continue
    user_input = input("\n=========================\nChoose a model to try another sentence:\n\tSeq2Seq (COWS-L2H only): 1\n\tSeq2Seq (COWS-L2H + Synthetic): 2\n\tToken-Classification (COWS-L2H only): 3\n\tToken-Classification (COWS-L2H + Synthetic): 4\nOr type exit to leave.\n")

