'''
File: generate.py
Author: Amber Converse
Purpose: This script generates synthetic errorful sentences from well-formed Spanish sentences in a corpus.
'''

import re
import random
import json
import argparse
from pathlib import Path
import spacy

from utils.utils import load_morpho_dict, load_vocab, create_vocab_index, get_path, follow_path, apply_labels, mutate
from label import label_sentence

from utils.utils import UPOS_TO_SIMPLE, AVAILABLE_TYPES, HIERARCHY_DEF, DEFAULT_PATH_FOR_POS

'''
Definition of Error Dictionary
{
    The type of error to be introduced:
        - MIXUP is a mixup of 2 or more words that are commonly mismatched
        - SWAP is a swapping of 2 words of specific criteria
        - MORPH is a change in morphology of a specific type of word (e.g. VERB, ARTICLE)
        - ADD is an addition of an erroneous word or set of words
        - DELETE deletes a random word
        - REPLACE replaces a random word with another random word
    "type": "MIXUP"|"SWAP"|"MORPH"|"ADD"|"DELETE", 

    criteria and criteria_pos are used to determine if a word or set of words can have the error
    apply to them. criteria is a regex string to be used to match with the lemma of the word being
    checked. For example, if the regex is r"^(estar|ser)$", it defines that this error can be applied to
    any form of estar or ser (e.g. soy, estuviera, fuera). criteria_pos is optional, but if present the
    word(s) must be of the given part(s) of speech.
    "crtieria": r"...",
    "criteria_pos": [PART_OF_SPEECH, ...],

    If True, code will attempt to match the morphology of the input word (generally used for MIXUP)
    "match_morph": True|False,

    Used for randomization of errors, higher weight means the error will appear more often in the corpus.
    "weight": int,

    The available output lemmas for this error, e.g. ["ser", "estar"] means this error can result in words
    that are forms of ser and estar.
    "output": [LEMMA, ...]

    This part is for MORPH errors. Defines which types of mutations can be applied.
    "mutate_categories": [MUTATE_CATEGORY, ...]
}
'''
def fill_empty_fields(error):
    '''
    This function ensures an error dict has all fields filled.
    '''
    if "type" not in error:
        error["type"] = "MISSING TYPE"
    if "criteria" not in error:
        error["criteria"] = r"*"
    if "criteria_pos" not in error:
        error["criteria_pos"] = []
    if "output" not in error:
        error["output"] = []
    if "weight" not in error:
        error["weight"] = 1
    if "match_morph" not in error:
        error["match_morph"] = False
    if "mutate_categories" not in error:
        error["mutate_categories"] = []

def load_errors(error_files):
    errors = []
    for error_file in error_files:
        with open(error_file, 'r') as f:
            errors += [fill_empty_fields(error) for error in json.load(f)]
    return errors

def token_list_to_str(sentence):
    sentence_str = ""
    first_token = True
    no_space = False
    for token in sentence:
        if re.match("^[\.\,\(\)\{\}\?\¿\!\¡]$", token.text) or first_token or no_space:
            sentence_str += token.text
            first_token = False
            no_space = False
        else:
            sentence_str += " " + token.text
        
        if re.match("^[\¿\¡]$", token.text): # Handle beginning punctuation
            no_space = True
    
    if sentence_str.endswith("EOS"): # Remove end token
        sentence_str = sentence_str[:-4]

    return sentence_str

def can_apply(error, sentence):
    num_valid = 0
    for token in sentence:
        if re.match(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
            num_valid += 1
            if not error["type"] == "swap" or num_valid > 1: # Handle the fact that swap needs two words of the criteria
                return True
    return False

def apply_error(error, sentence, lemma_to_morph, vocab, nlp):
    new_sentence = []
    if error["type"] == "mixup":
        choices = []
        for i, token in enumerate(sentence):
            if re.match(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        mixup_i = random.choice(choices)
        new_sentence = []
        for i, token in sentence:
            if i == mixup_i:
                possible_lemmas = error["output"].copy()
                possible_lemmas.remove(token.lemma_)
                new_lemma = random.choice(possible_lemmas)
                if error["match_morph"]:
                    cur_path = get_path(token)
                    new_token = follow_path(lemma_to_morph[new_lemma], cur_path)
                else:
                    try:
                        new_token = follow_path(lemma_to_morph[new_lemma], DEFAULT_PATH_FOR_POS[UPOS_TO_SIMPLE[token.pos_]])
                    except KeyError:
                        print(f"KeyError trying to get surface form of {new_lemma}, using lemma...")
                        new_token = new_lemma
                new_sentence.append(new_token[0].upper() + new_token[1:] if token.text[0].isupper() else new_token)
            else:
                new_sentence.append(token)
    elif error["type"] == "swap":
        choices = []
        for i, token in enumerate(sentence):
            if re.match(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        new_sentence = sentence.copy()
        morph_i = random.choice(choices)
        choices.remove(morph_i)
        if len(choices) == 0:
            print("Unable to swap, only 1 word in sentence meets criteria. No change to sentence.")
        else:
            morph_j = random.choice(choices)
            new_sentence[morph_i], new_sentence[morph_j] = new_sentence[morph_j], new_sentence[morph_i]
    elif error["type"] == "morph":
        choices = []
        for i, token in enumerate(sentence):
            if re.match(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        morph_i = random.choice(choices)
        new_sentence = []
        for i, token in sentence:
            if i == morph_i:
                category_to_mutate = random.choice(error["mutate_categories"])
                if category_to_mutate == "CAPITALIZE":
                    new_sentence.append(token.text[0].lower() + token.text[1:] if token.text[0].issupper() else token.text[0].upper() + token.text[1:])
                else:
                    label_param = category_to_mutate # First part of label param

                    cur_path = get_path(token)
                    index_of_change = HIERARCHY_DEF[label_param.split('-')[0]]
                    if isinstance(index_of_change, list): # Special case for verbs when changing mood
                        if cur_path[2] == "SURFACE_FORM": # Special verb case
                            index_of_change = 1
                        else: # Normal verb case
                            index_of_change = 2
                    cur_category = cur_path[index_of_change]

                    possible_categories = AVAILABLE_TYPES[category_to_mutate].copy()
                    possible_categories.remove(cur_category)
                    label_param += "-" + random.choice(possible_categories)
                    new_token = mutate(token, label_param, lemma_to_morph, nlp)
                    new_sentence.append(new_token[0].upper() + new_token[1:] if new_token[0].issupper() else new_token)
            else:
                new_sentence.append(token)
    elif error["type"] == "add":
        choices = []
        for i, token in enumerate(sentence):
            if re.match(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        add_i = random.choice(choices)
        if error["output"]:
            add_words = error["output"]
        else: # If no list of possible replacements, just choose any random word
            add_words = vocab
        new_sentence = []
        for i, token in enumerate(sentence):
            if i == add_i:
                new_word = random.choice(add_words)
                if random.randint(0,1):
                    new_sentence.append(new_word[0].upper() + new_word[1:] if token.text[0].issupper() else new_word)
                    new_sentence.append(token.text[0].lower() + token.text[1:] if token.text[0].issupper() else token)
                else:
                    new_sentence.append(token)
                    new_sentence.append(token.text[0].lower() + token.text[1:] if token.text[0].issupper() else token)
            else:
                new_sentence.append(token)
    elif error["type"] == "delete":
        choices = []
        for i, token in enumerate(sentence):
            if re.match(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        delete_i = random.choices(choices)
        new_sentence = [token for i, token in enumerate(sentence) if i != delete_i]
    elif error["type"] == "replace":
        choices = []
        for i, token in enumerate(sentence):
            if re.match(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        replace_i = random.choice(choices)
        if error["output"]:
            replace_words = error["output"]
        else: # If no list of possible replacements, just choose any random word
            replace_words = vocab
        new_word = random.choice(replace_words)
        new_sentence = [token if i != replace_i else new_word[0].upper() + new_word[1:] if token.text[0].issupper() else new_word
                        for i, token in enumerate(sentence)]
    else:
        print(f"Unknown error type {error["type"]}, no change to sentence.")
        new_sentence = sentence
    
    return nlp(' '.join([str(token) for token in new_sentence])) # Re-apply nlp with change

def generate_errorful_sentences(input_file, output_file, errors,
                                lemma_to_morph, vocab,
                                max_error=3, num_sentences=1,
                                include_token_labels=True, verify=False):
    '''
    '''
    nlp = spacy.load("es_dep_news_trf")
    vocab_index = create_vocab_index(vocab)

    # Create list of words in the vocabulary that are not special tokens or morphological parts
    vocab_only_words = [word for word in vocab if not re.match(r"^(\[.*\])|(#.*)$", word)]

    errorful_sentences = []
    with open(input_file, 'r') as f:
        for line in f:
            sentence = nlp(line)
            for _ in range(num_sentences):
                num_errors = random.randint(0, max_error)
                cur_sentence = [token for token in sentence] # Convert to list of tokens
                for _ in range(num_errors):
                    choices = []
                    for i, error in enumerate(errors):
                        if can_apply(error, cur_sentence):
                            choices += [i] * error["weight"]
                    new_sentence = apply_error(errors[random.choice(choices)], cur_sentence, lemma_to_morph, vocab_only_words, nlp)
                    cur_sentence = [token for token in new_sentence] # Update sentence
                
                errorful_sentences.append((token_list_to_str(cur_sentence), token_list_to_str(sentence)))

    with open(output_file, 'w') as f:
        for sentence in errorful_sentences:
            f.write(sentence[0] + '\n')
            if include_token_labels:
                token_labels = label_sentence(sentence[0], sentence[1], lemma_to_morph, vocab_index, nlp)
                if verify:
                    decoded_sentence = apply_labels(nlp(sentence[0]), token_labels.split('\t'), lemma_to_morph, vocab, nlp)
                    if decoded_sentence != sentence[1]:
                        print(f"VERIFY FAILED!\nReport:\n\tErrorful Sentence:{sentence[0]}\n\tGenerated Labels:{token_labels}\n\tTarget:{sentence[1]}\n\tResult from Decode:{decoded_sentence}")
                f.write(token_labels + '\n')
            f.write(sentence[1] + "\n\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to a file with sentences in Spanish on each line")
    parser.add_argument("output_file", help="output path")
    parser.add_argument("error_files", nargs="+", help="error file paths")
    parser.add_argument("-m", "--max_error", default=3, help="The maximum number of errors that can be generated in a sentence. By default 3")

    parser.add_argument("--dict_file", default="lang_def/morpho_dict.json", help="path to the dictionary file which supplies different morphological forms for a word")
    parser.add_argument("--vocab_file", default="lang_def/vocab.txt", help="path to the vocab file containing all words in your model's vocabulary")

    parser.add_argument("--seed", default=42, type=int, help="Seed for random error generation")

    parser.add_argument("-n", "--num-sentences", default=1, help="the number of errorful sentences that will be generated from each correct sentence in the supplied corpus. By default 1")
    parser.add_argument("-t", "--token",
                        help="the output synthetic data will include token-level labels",
                        action="store_true")
    parser.add_argument("-v", "--verify",
                        help="generated token labels will be verified as correct by running the decode algorithm",
                        action="store_true")

    args = parser.parse_args()

    input_file = args.input_file
    if not args.output_file:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_synth{input_path.suffix}"
    else:
        output_file = args.output_file

    errors = load_errors(args.error_files)

    random.seed(args.seed)

    generate_errorful_sentences(input_file, output_file, errors,
                                load_morpho_dict(args.dict_file), load_vocab(args.vocab_file),
                                args.max_error, args.num_sentences,
                                token=args.token, verify=args.verify)