import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertConfig, BertModel, AutoTokenizer, AutoConfig, PreTrainedModel

from utils.utils import labels_to_vec, vec_to_labels, apply_labels

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
