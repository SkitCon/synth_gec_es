'''
File: generate.py
Author: Amber Converse
Purpose: This script generates synthetic errorful sentences from well-formed Spanish sentences in a corpus.
'''

import re
import random
import json
import time
import argparse
import warnings
import traceback
from pathlib import Path
import spacy

from utils.utils import load_morpho_dict, load_vocab, create_vocab_index, get_path, follow_path, apply_labels, mutate, clean_text, load_modified_nlp, parallelize_function
from label import label_sentence

from utils.utils import UPOS_TO_SIMPLE, AVAILABLE_TYPES, HIERARCHY_DEF, DEFAULT_PATH_FOR_POS

from utils.custom_errors import FailedToMeetStrictRequirementException

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
    return error

def load_errors(error_files):
    errors = []
    for error_file in error_files:
        with open(error_file, 'r') as f:
            errors += [fill_empty_fields(error) for error in json.load(f)]
    return errors

def token_list_to_str(token_list):
    sentence = ""
    first_token = True
    no_space = False
    for token in token_list:
        if not isinstance(token, list):
            token = [token]
        for cur_token in token:
            if re.match("^[\.\,\?\!\)\}\]\-\:\;]$", str(cur_token)) or first_token or no_space:
                sentence += str(cur_token)
                first_token = False
                no_space = False
            elif re.match("^\#\#.*$", str(cur_token)): # Sub-word
                sentence += str(cur_token)[2:]
                first_token = False
                no_space = False
            else:
                sentence += " " + str(cur_token)
            
            if re.match("^[\¿\¡\(\{\[\"]$", str(cur_token)): # Handle beginning punctuation
                no_space = True
    
    if sentence.endswith("E"): # Remove end token
        sentence = sentence[:-2]
    
    return sentence

def can_apply(error, sentence):
    num_valid = 0
    for token in sentence:
        if re.fullmatch(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
            num_valid += 1
            if not error["type"] == "swap" or num_valid > 1: # Handle the fact that swap needs two words of the criteria
                return True
    return False

def apply_error(error, sentence, lemma_to_morph, vocab, nlp, verbose=False, silence_warnings=False):
    new_sentence = []

    if verbose:
        print(f"Applying error: {error['name']}")
        print(f"To sentence: {sentence}")
    if error["type"] == "mixup":
        choices = []
        for i, token in enumerate(sentence):
            if re.fullmatch(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        if len(choices) != 0:
            mixup_i = random.choice(choices)
        else:
            if not silence_warnings:
                print(f"No valid choices for mixup error: {error['name']}")
            mixup_i = -1
        new_sentence = []
        if verbose:
            print(f"Possible choices: {choices}")
            print(f"Applying error to: {mixup_i}")
        for i, token in enumerate(sentence):
            if i == mixup_i:
                possible_lemmas = error["output"].copy()
                possible_lemmas.remove(token.lemma_)
                if len(choices) != 0:
                    new_lemma = random.choice(possible_lemmas)
                else:
                    if not silence_warnings:
                        print(f"No possible lemmas to replace with for {token.lemma_} for mixup error: {error['name']}")
                    new_lemma = token.lemma_
                if error["match_morph"]:
                    cur_path = get_path(token)
                    try:
                        new_token = follow_path(lemma_to_morph[new_lemma], cur_path)
                    except KeyError as e:
                        print(f"KeyError {e} while trying to match morphology. Trying default path.")
                        try:
                            new_token = follow_path(lemma_to_morph[new_lemma], DEFAULT_PATH_FOR_POS[UPOS_TO_SIMPLE[token.pos_]])
                        except KeyError as e:
                            print(f"KeyError {e} while trying to match DEFAULT morphology. Using lemma.")
                            new_token = token.lemma_
                else:
                    try:
                        new_token = follow_path(lemma_to_morph[new_lemma], DEFAULT_PATH_FOR_POS[UPOS_TO_SIMPLE[token.pos_]])
                    except KeyError:
                        if not silence_warnings and verbose:
                            print(f"KeyError trying to get surface form of {new_lemma}, using lemma...")
                        new_token = new_lemma
                new_sentence.append(new_token[0].upper() + new_token[1:] if token.text[0].isupper() else new_token)
            else:
                new_sentence.append(token)
    elif error["type"] == "swap":
        choices = []
        for i, token in enumerate(sentence):
            if re.fullmatch(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        new_sentence = sentence.copy()
        if len(choices) != 0:
            if verbose:
                print(f"Possible choices for i: {choices}")
            swap_i = random.choice(choices)
            choices.remove(swap_i)
            if len(choices) == 0:
                print("Unable to swap, only 1 word in sentence meets criteria. No change to sentence.")
            else:
                if verbose:
                    print(f"Possible choices for j: {choices}")
                swap_j = random.choice(choices)
                if verbose:
                    print(f"Applying error to: {swap_i} and {swap_j}")
                new_sentence[swap_i], new_sentence[swap_j] = new_sentence[swap_j], new_sentence[swap_i]
        else:
            if not silence_warnings:
                print(f"No valid choices for swap error: {error['name']}")
    elif error["type"] == "morph":
        choices = []
        for i, token in enumerate(sentence):
            if re.fullmatch(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        if len(choices) != 0:
            morph_i = random.choice(choices)
        else:
            if not silence_warnings:
                print(f"No valid choices for morph error: {error['name']}")
            morph_i = -1
        if verbose:
            print(f"Possible choices: {choices}")
            print(f"Applying error to: {morph_i}")
        new_sentence = []
        for i, token in enumerate(sentence):
            if i == morph_i:
                category_to_mutate = random.choice(error["mutate_categories"])
                if category_to_mutate == "CAPITALIZE":
                    new_sentence.append(token.text[0].lower() + token.text[1:] if token.text[0].isupper() else token.text[0].upper() + token.text[1:])
                else:
                    label_param = category_to_mutate # First part of label param

                    cur_path = get_path(token)
                    if len(cur_path) == 3 and UPOS_TO_SIMPLE[token.pos_] == "VERB": # Special verb
                        cur_path = DEFAULT_PATH_FOR_POS["VERB"]
                    index_of_change = HIERARCHY_DEF[label_param.split('-')[0]]
                    if isinstance(index_of_change, list): # Special case for verbs when changing mood
                        if cur_path[2] == "SURFACE_FORM": # Special verb case
                            index_of_change = 1
                        else: # Normal verb case
                            index_of_change = 2
                    cur_category = cur_path[index_of_change]

                    possible_categories = AVAILABLE_TYPES[category_to_mutate].copy()
                    if not cur_category in possible_categories:
                        new_token = str(token)
                    else:
                        possible_categories.remove(cur_category)
                        label_param += "-" + random.choice(possible_categories)
                        try:
                            new_token = mutate(token, label_param, lemma_to_morph, nlp)
                        except KeyError:
                            if not silence_warnings and verbose:
                                print(f"Tried to mutate {token}, but was not in morphology dict. No change to sentence.")
                            new_token = str(token)
                    if type(new_token) == float:
                        if not silence_warnings:
                            print(f"{new_token} was not a string. Converting to string.")
                        new_token = str(new_token)
                    new_sentence.append(new_token[0].upper() + new_token[1:] if new_token[0].isupper() else new_token)
            else:
                new_sentence.append(token)
    elif error["type"] == "add":
        choices = []
        for i, token in enumerate(sentence):
            if re.fullmatch(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        if len(choices) != 0:
            add_i = random.choice(choices)
        else:
            if not silence_warnings:
                print(f"No valid choices for add error: {error['name']}")
            add_i = -1
        if verbose:
            print(f"Possible choices for where to add: {choices}")
            print(f"Applying error to: {add_i}")
        if error["output"]:
            add_words = error["output"]
        else: # If no list of possible replacements, just choose any random word
            add_words = vocab
        new_sentence = []
        if verbose:
            print(f"Number of possible additions: {len(add_words)}")
        for i, token in enumerate(sentence):
            if i == add_i:
                new_word = random.choice(add_words)
                if random.randint(0,1):
                    if verbose:
                        print(f"Adding {new_word} AFTER {add_i}")
                    new_sentence.append(new_word[0].upper() + new_word[1:] if token.text[0].isupper() else new_word)
                    new_sentence.append(token.text[0].lower() + token.text[1:] if token.text[0].isupper() else token)
                else:
                    if verbose:
                        print(f"Adding {new_word} AFTER {add_i}")
                    new_sentence.append(token)
                    new_sentence.append(token.text[0].lower() + token.text[1:] if token.text[0].isupper() else token)
            else:
                new_sentence.append(token)
    elif error["type"] == "delete":
        choices = []
        for i, token in enumerate(sentence):
            if re.fullmatch(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        if len(choices) != 0:
            delete_i = random.choice(choices)
        else:
            if not silence_warnings:
                print(f"No valid choices for delete error: {error['name']}")
            delete_i = -1
        if verbose:
            print(f"Possible choices for where to delete: {choices}")
            print(f"Applying error to: {delete_i}")
        new_sentence = [token for i, token in enumerate(sentence) if i != delete_i]
    elif error["type"] == "replace":
        choices = []
        for i, token in enumerate(sentence):
            if re.fullmatch(error["criteria"], token.lemma_) and (UPOS_TO_SIMPLE[token.pos_] in error["criteria_pos"] or not error["criteria_pos"]):
                choices.append(i)
        if len(choices) != 0:
            replace_i = random.choice(choices)
        else:
            if not silence_warnings:
                print(f"No valid choices for replace error: {error['name']}")
            replace_i = -1
        if verbose:
            print(f"Possible choices for where to replace: {choices}")
            print(f"Applying error to: {replace_i}")
        if error["output"]:
            replace_words = error["output"]
        else: # If no list of possible replacements, just choose any random word
            replace_words = vocab
        new_word = random.choice(replace_words)
        if verbose:
            print(f"Number of possible replacements: {len(replace_words)}")
            print(f"Replacing with word at {replace_i} with {new_word}")
        new_sentence = [token if i != replace_i else new_word[0].upper() + new_word[1:] if token.text[0].isupper() else new_word
                        for i, token in enumerate(sentence)]
    else:
        print(f"Unknown error type {error['type']}, no change to sentence.")
        new_sentence = sentence
    
    if verbose:
        print(f"New sentence: {token_list_to_str(new_sentence)}")
    return nlp(token_list_to_str(new_sentence)) # Re-apply nlp with change

def label_sentence_error_wrapper(sentence_pair, lemma_to_morph, vocab_index, verbose=False, silence_warnings=False, strict=False):
    global nlp
    errorful = sentence_pair[0]
    correct = sentence_pair[1]
    try:
        return label_sentence(errorful, correct, lemma_to_morph, vocab_index, nlp, verbose, silence_warnings, strict)
    except FailedToMeetStrictRequirementException as e:
        if not silence_warnings:
            print(f"Failed to generate labels due to strict requirements: {e}\n\tErrorful sentence: {errorful}\n\tCorrect sentence: {correct}")
        return ""
    except Exception as e:
        if not silence_warnings:
            print(f"Failed to generate labels due to {e}\n\tErrorful sentence: {errorful}\n\tCorrect sentence: {correct}")
            print(traceback.format_exc())
        return ""
    
def apply_labels_error_wrapper(sentence_label, lemma_to_morph, vocab, verbose=False, silence_warnings=False):

    # Ok, I really don't like this as it requires this function to be declared in the file where nlp is loaded, but I need
    # it because the nlp object is unpickleable, so it can't be an argument in a function passed to pool.map
    global nlp

    errorful_sentence = sentence_label[0]
    labels = sentence_label[1]

    try:
        tokenized_errorful_sentence = nlp(errorful_sentence)
        if verbose:
            print(f"=================================\nVerifying sentence: {errorful_sentence}\nTokenization: {tokenized_errorful_sentence}")
        decoded_sentence = apply_labels(tokenized_errorful_sentence, labels.split('\t'), lemma_to_morph, vocab, nlp)
    except KeyError as e:
        if not silence_warnings:
            print(f"VERIFY FAILED!!!\n\tCaused by KeyError: {e}\n\tReport:\n\tErrorful Sentence:{errorful_sentence}\n\tGenerated Labels:{labels}")
        return ""
    except IndexError as e:
        if not silence_warnings:
            print(f"VERIFY FAILED!!!\n\tCaused by IndexError: {e}\n\tReport:\n\tErrorful Sentence:{errorful_sentence}\n\tGenerated Labels:{labels}")
            print(traceback.format_exc())
        return ""
    return decoded_sentence

def generate_errorful_sentences(correct_sentence, errors, lemma_to_morph, vocab_only_words, min_error=0, max_error=3, num_sentences=1, verbose=False, silence_warnings=False):
    global nlp

    if re.sub(r"\s", '', correct_sentence) == "": # Empty line
        return []
    
    correct_sentence = nlp(clean_text(correct_sentence))

    sentence_pairs = []
    if verbose:
        print(f"=========================\nBeginning error generation for sentence: {correct_sentence}")
    for _ in range(num_sentences):
        num_errors = random.randint(min_error, max_error)
        cur_sentence = [token for token in correct_sentence] # Convert to list of tokens
        if verbose:
            print(f"--------------------\nGeneratng errorful sentence with {num_errors} errors.")
        for _ in range(num_errors):
            choices = []
            for i, error in enumerate(errors):
                if can_apply(error, cur_sentence):
                    choices += [i] * error["weight"]
            if len(choices) != 0:
                new_sentence = apply_error(errors[random.choice(choices)], cur_sentence, lemma_to_morph, vocab_only_words, nlp, verbose=verbose, silence_warnings=silence_warnings)
                cur_sentence = [token for token in new_sentence] # Update sentence
            else:
                print("No valid errors for sentence (empty sentence?). No errors generated.")
        sentence_pairs.append((token_list_to_str(cur_sentence), token_list_to_str(correct_sentence)))
    return sentence_pairs
    
def main(input_file, output_file, errors,
         lemma_to_morph, vocab, spacy_model="es_dep_news_trf", tokenizer_model="dccuchile/bert-base-spanish-wwm-cased",
         min_error=0, max_error=3, num_sentences=1,
         include_token_labels=True, verify=False, verbose=False,
         silence_warnings=False, strict=False, n_cores=1):
    
    global nlp
    nlp = load_modified_nlp(model_path=spacy_model, tokenizer_path=tokenizer_model)
    vocab_index = create_vocab_index(vocab)

    # Create list of words in the vocabulary that are not special tokens or morphological parts
    vocab_only_words = [word for word in vocab if not re.match(r"^(\[.*\])|(#.*)$", word)]

    sentence_pairs = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        sentences = [lines[i] for i in range(0, len(lines), 2) if lines[i].strip() != ""]
        for i in range(0, len(sentences), n_cores):

            slice = sentences[i:min(i+n_cores, len(sentences))]

            print(f"================================\nGenerating {len(slice)*num_sentences} errorful sentences from {len(slice)} correct sentences...")
            start_time = time.time()
            sentence_pair_groups = parallelize_function(slice, generate_errorful_sentences, n_cores,
                                                        kwargs={"errors":errors, "lemma_to_morph":lemma_to_morph, "vocab_only_words":vocab_only_words,
                                                                "min_error":min_error, "max_error":max_error, "num_sentences":num_sentences, "verbose":verbose, "silence_warnings":silence_warnings})
            cur_sentence_pairs = [sentence_pair for group in sentence_pair_groups for sentence_pair in group]
            sentence_pairs += cur_sentence_pairs
            cur_time = time.time()
            print(f"Succesfully generated {len(cur_sentence_pairs)} errorful sentences in {round(cur_time-start_time, 2)} seconds.\nAverage time per sentence: {round((cur_time-start_time) / len(cur_sentence_pairs), 3)}")
            print(f"{round(min(i+n_cores, len(sentences)) / len(sentences) * 100, 1)}% done.")

    included_sentences = 0
    excluded_sentences = 0
    with open(output_file, 'w') as f:
        for i in range(0, len(sentence_pairs), n_cores):

            slice = sentence_pairs[i:min(i+n_cores, len(sentence_pairs))]

            if include_token_labels:
                print(f"================================\nGenerating labels for {len(slice)} sentences...")
                start_time = time.time()
                labels = parallelize_function(slice, label_sentence_error_wrapper, n_cores, kwargs={"lemma_to_morph":lemma_to_morph, "vocab_index":vocab_index,
                                                                                                    "verbose":verbose, "silence_warnings":silence_warnings, "strict":strict})
                excluded_sentences += len([label for label in labels if len(label) == 0])
                cur_time = time.time()
                print(f"Completed labeling for {len(labels)} sentences in {round(cur_time - start_time, 2)} seconds.\n{len([label for label in labels if len(label) == 0])} could not be labeled.\nAverage {round((cur_time - start_time) / n_cores, 3)} seconds per sentence")

                if verify:
                    print(f"===================================\nVerifying {len(labels)} labels...")
                    start_time = time.time()
                    decoded_sentences = parallelize_function(zip(list(zip(*slice))[0], labels), apply_labels_error_wrapper, n_cores, kwargs={"lemma_to_morph":lemma_to_morph, "vocab":vocab,
                                                                                                                                             "verbose":verbose, "silence_warnings":silence_warnings})
                
                failed_verification = 0
                for j in range(len(slice)):
                    if verify and slice[j][1] != decoded_sentences[j]:
                        if labels[j] != "" and decoded_sentences[j] != "": # No need to warn if this sentence already failed labeling or errored during apply_labels
                            if not silence_warnings:
                                print(f"VERIFY FAILED!\nReport:\n\tErrorful Sentence:{slice[j][0]}\n\tGenerated Labels:{labels[j]}\n\tTarget:{slice[j][1]}\n\tResult from Decode:{decoded_sentences[j]}")
                        failed_verification += 1
                        excluded_sentences += 1
                        continue
                    else:
                        f.write(f"{slice[j][0]}\n{labels[j]}\n{slice[j][1]}\n\n")
                        included_sentences += 1
                cur_time = time.time()
                if verify:
                    print(f"Completed verification for {len(decoded_sentences)} sentences in {round(cur_time - start_time, 2)} seconds.\n{failed_verification} sentences failed verification.\nAverage {round((cur_time - start_time) / n_cores, 3)} seconds per sentence")
                    print(f"{round(min(i+n_cores, len(sentence_pairs)) / len(sentence_pairs) * 100, 1)}% done.")
            else:
                for j in range(len(slice)):
                    f.write(f"{slice[j][0]}\n{slice[j][1]}\n\n")
                    included_sentences += 1

    print(f"===============================\nExcluded {excluded_sentences} sentences.")
    print(f"Saved {included_sentences} sentences.")
    
if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to a file with sentences in Spanish on each line")
    parser.add_argument("output_file", help="output path")
    parser.add_argument("--error_files", nargs="+",
                        default= ["lang_def/errors/mixup_errors.json", "lang_def/errors/swap_errors.json",
                                  "lang_def/errors/morph_errors.json", "lang_def/errors/add_errors.json",
                                  "lang_def/errors/delete_errors.json", "lang_def/errors/replace_errors.json"],
                        help="error file paths")
    parser.add_argument("-min", "--min_error", default=0, type=int, help="The minimum number of errors that can be generated in a sentence. By default 0")
    parser.add_argument("-max", "--max_error", default=3, type=int, help="The maximum number of errors that can be generated in a sentence. By default 3")

    parser.add_argument("--dict_file", default="lang_def/morpho_dict_updated.json", help="path to the dictionary file which supplies different morphological forms for a word")
    parser.add_argument("--vocab_file", default="lang_def/vocab.txt", help="path to the vocab file containing all words in your model's vocabulary")
    parser.add_argument("--spacy_model", default="es_dep_news_trf", help="spaCy model to use")
    parser.add_argument("--tokenizer_model", default="dccuchile/bert-base-spanish-wwm-cased", help="Tokenizer model to use (local or HuggingFace path)")

    parser.add_argument("--n_cores", default=1, type=int, help="Number of cores to use (1 for no multi-processing)")
    parser.add_argument("--seed", default=42, type=int, help="Seed for random error generation")

    parser.add_argument("-n", "--num_sentences", type=int, default=1, help="the number of errorful sentences that will be generated from each correct sentence in the supplied corpus. By default 1")
    parser.add_argument("-t", "--token",
                        help="the output synthetic data will include token-level labels",
                        action="store_true")
    parser.add_argument("--verify",
                        help="generated token labels will be verified as correct by running the decode algorithm",
                        action="store_true")
    parser.add_argument("-v", "--verbose",
                        help="Print debugging statements",
                        action="store_true")
    parser.add_argument("-sw", "--silence_warnings",
                        help="Silence warnings",
                        action="store_true")
    parser.add_argument("--strict",
                        help="Be strict about which sentences are included in the output file. This has the primary effect of excluding sentences where a MUTATE verification failed and a REPLACE had to be used to repair the labels.",
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

    main(input_file, output_file, errors, load_morpho_dict(args.dict_file), load_vocab(args.vocab_file),
         spacy_model=args.spacy_model, tokenizer_model=args.tokenizer_model,
         min_error=args.min_error, max_error=args.max_error, num_sentences=args.num_sentences,
         include_token_labels=args.token, verify=args.verify, verbose=args.verbose,
         silence_warnings=args.silence_warnings, strict=args.strict, n_cores=args.n_cores)