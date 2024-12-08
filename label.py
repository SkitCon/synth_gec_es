'''
File: label.py
Author: Amber Converse
Purpose: This script takes errorful sentences + target sentences and translates them into token-level
    edits using shortest edit distance.

    Args are:

    * input file is a path to a file with an errorful sentence on one line, the target sentence on the next,
        and a blank line before the next sentence
    * output file is optional, defines the output path to place the labeled sentences in. If no output path
        is supplied, it is placed in the same directory as the input file w/ the name [input file name]_labeled.txt.
    * --dict_file is the path to the dictionary file which supplies different morphological forms
        for a word
    * --vocab_file is the path to the vocab file containing all words in your model's vocabulary
'''

import re
import time
import json
import argparse
import warnings
from pathlib import Path
import numpy as np
import traceback
import spacy
from bs4 import BeautifulSoup
from transformers import BertTokenizer
from unidecode import unidecode


from utils.utils import load_morpho_dict, load_vocab, get_path, get_new_path, create_vocab_index, apply_labels, mutate, CONTEXT_WINDOW, process_lemma, load_modified_nlp, restore_nlp, clean_text, parallelize_function

from utils.custom_errors import FailedToMeetStrictRequirementException

COST = {"KEEP": 0,
        "REPLACE": 1,
        "DELETE": 1,
        "ADD": 1}

CATEGORIES = {"NOUN": ["NUMBER", "GENDER"],
              "ADJ": ["NUMBER", "GENDER"],
              "ADV": ["NUMBER", "GENDER"],
              "PRONOUN": ["NUMBER", "GENDER", "CASE"],
              "PERSONAL_PRONOUN": ["NUMBER", "GENDER", "CASE", "PRONOUN_TYPE", "REFLEXIVE"],
              "ARTICLE": ["NUMBER", "GENDER", "DEFINITE"],
              "VERB": ["NUMBER", "MOOD", "TIME", "PERSON"]}

def add_entry_to_morphology(surface_form, path, dict):
    if len(path) == 0 or path[0] == "SURFACE_FORM":
        dict["SURFACE_FORM"] = surface_form
    else:
        if not path[0] in dict:
            dict[path[0]] = {}
        add_entry_to_morphology(surface_form, path[1:], dict[path[0]])

def generate_mutation_sequence(path):
    pos = path[0]
    if pos == "VERB" and path[1] in ["GER", "PAST-PART", "INF"]: # Special verb
        return ["<MUTATE param=\"POS-VERB\"/>", f"<MUTATE param=\"MOOD-{path[1]}\"/>"]
    else:
        return [f"<MUTATE param=\"POS-{path[0]}\"/>"] + [f"<MUTATE param=\"{CATEGORIES[pos][i-1]}-{path[i]}\"/>" for i in range(1, len(path)-1)]

def link_with_mutation(errorful_token, correct_token):

    # Figure out series of mutations to get from errorful token to correct token
    errorful_path = get_path(errorful_token)
    correct_path = get_path(correct_token)

    errorful_sequence = generate_mutation_sequence(errorful_path)
    correct_sequence = generate_mutation_sequence(correct_path)

    # Eliminate redundant labels
    final_sequence = []
    for i in range(len(errorful_sequence) + 1):
        if i >= len(errorful_sequence) and i < len(correct_sequence):
            final_sequence += correct_sequence[i:]
        elif i >= len(errorful_sequence) and i >= len(correct_sequence):
            break
        elif i >= len(correct_sequence) and i < len(errorful_sequence):
            final_sequence = correct_sequence
            break
        elif errorful_sequence[i] != correct_sequence[i]:
            final_sequence.append(correct_sequence[i])

    if correct_token.text[0].isupper(): # Check if capitalization needs to be applied
        final_sequence.append("<MUTATE param=\"CAPITALIZE-TRUE\"/>")
    elif correct_token.text[0].islower() and not errorful_token.text[0].islower(): # Check if capitalization needs to be removed
        final_sequence.append("<MUTATE param=\"CAPITALIZE-FALSE\"/>")

    return final_sequence

def verify_mutation(token, sentence, token_idx, labels, lemma_to_morph, nlp, correct_token=None, silence_warnings=False):
    token = [token]
    for label in labels:
        if "MUTATE" in label:
            
            if not isinstance(token, list):
                token = [token]

            param = BeautifulSoup(label, features="html.parser").find_all(True)[0].get("param", "").upper()
            try:
                token = [mutate(token[0], param, lemma_to_morph, nlp)] + token[1:]
            except KeyError:
                if correct_token:
                    # Add entry to dict
                    new_path = [process_lemma(token[0].lemma_)] + get_new_path(correct_token, param)
                    if not silence_warnings:
                        print(f"Adding {correct_token.text.lower()} to dict at {new_path}")
                    add_entry_to_morphology(correct_token.text.lower(), new_path, lemma_to_morph)
                    # Try again
                    token = [mutate(token[0], param, lemma_to_morph, nlp)]
                else:
                    # If no correct token, 
                    raise KeyError
            token = restore_nlp(sentence, token, token_idx, nlp)
    if len(token) > 1:
        all_strs = [str(cur_token) for cur_token in token]
        combined = []
        for cur_token in all_strs:
            if cur_token.startswith("##") and len(combined) > 0: # Tokenized as parts, needs to be combined with previous word
                combined[-1] += cur_token[2:]
            else:
                combined.append(cur_token)
        return ' '.join(combined)
    else:
        return token[0]

def label_sentence(errorful, correct, lemma_to_morph, vocab_index, nlp, verbose=False, silence_warnings=False, strict=False):

    errorful_doc = nlp(errorful)
    correct_doc = nlp(correct)

    if verbose:
        print(f"=================================\nGenerating labels for:\n{errorful}\nto\n{correct}")
        print(f"Tokenization:\n\tErrorful: {errorful_doc}\n\tCorrect: {correct_doc}")

    dp = np.zeros((len(errorful_doc)+1, len(correct_doc)+1))

    # Fill in base case
    dp[:, 0] = list(range(len(errorful_doc)+1)) * COST["DELETE"]
    dp[0, :] = list(range(len(correct_doc)+1)) * COST["ADD"]

    for i in range(1, len(errorful_doc)+1):
        for j in range(1, len(correct_doc)+1):
            if errorful_doc[i-1].text == correct_doc[j-1].text:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + COST["KEEP"], # KEEP
                    dp[i - 1][j] + COST["DELETE"],   # DELETE
                    dp[i][j - 1] + COST["ADD"]       # ADD
                )
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + COST["REPLACE"], # REPLACE
                    dp[i - 1][j] + COST["DELETE"],      # DELETE
                    dp[i][j - 1] + COST["ADD"]          # ADD
                )
                
    # Backtrace
    i = len(errorful_doc)
    j = len(correct_doc)
    operations = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and errorful_doc[i-1].text == correct_doc[j-1].text:
            if dp[i][j] == dp[i - 1][j - 1] + COST["KEEP"]:
                operations.append("KEEP")
                i -= 1
                j -= 1
                continue
        
        if j > 0 and dp[i][j] == dp[i][j - 1] + COST["ADD"]:
            operations.append("ADD")
            j -= 1
            continue
        
        if i > 0 and j > 0:
            if dp[i][j] == dp[i - 1][j - 1] + COST["REPLACE"]:
                operations.append("REPLACE")
                i -= 1
                j -= 1
                continue
        
        if i > 0 and dp[i][j] == dp[i - 1][j] + COST["DELETE"]:
            operations.append("DELETE")
            i -= 1
            continue
    
    operations.reverse()

    if verbose:
        print(f"Basic operations: {operations}")

    # Put all lemmas and tokens in a set to easily check if COPY or COPY+MUTATE is possible
    token_lemmas = {process_lemma(token.lemma_) for token in errorful_doc}
    tokens = {token.text for token in errorful_doc}

    # Translate basic operations to complex token labels
    labels = []
    cur_labels = []
    op_idx = 0
    errorful_idx = 0
    correct_idx = 0
    while op_idx < len(operations):

        operation = operations[op_idx]

        # Handle end of add sequence, ignore KEEP at the end of ADD sequence
        if operation == "KEEP" and len(cur_labels) > 0:
            labels.append(cur_labels)
            cur_labels = []
            op_idx += 1
            errorful_idx += 1
            correct_idx += 1
            continue
        if operation == "DELETE" and len(cur_labels) > 0:
            labels.append(cur_labels)
            cur_labels = []
            errorful_idx += 1
        # elif operation == "REPLACE" and len(cur_labels) > 0:
        #     correct_idx += 1
        
        if operation == "KEEP":
            cur_labels.append(["<KEEP/>"])
            labels.append(cur_labels)
            cur_labels = []
            op_idx += 1
            errorful_idx += 1
            correct_idx += 1
        elif operation == "DELETE":
            cur_labels.append(["<DELETE/>"])
            labels.append(cur_labels)
            cur_labels = []
            op_idx += 1
            errorful_idx += 1
        elif operation == "REPLACE":
            if process_lemma(errorful_doc[errorful_idx].lemma_) == process_lemma(correct_doc[correct_idx].lemma_): # Can mutate
                mutation_labels = link_with_mutation(errorful_doc[errorful_idx], correct_doc[correct_idx])
                
                # Verify mutation
                failed_to_resolve = True
                key_error = False
                try:
                    mutated_token = verify_mutation(errorful_doc[errorful_idx], [token for token in errorful_doc], errorful_idx, mutation_labels,
                                                    lemma_to_morph, nlp, correct_token=correct_doc[correct_idx], silence_warnings=silence_warnings)
                    if str(mutated_token) == correct_doc[correct_idx].text:
                        failed_to_resolve = False
                except KeyError as e:
                    key_error = True
                    error = e

                if failed_to_resolve:
                    if key_error:
                        msg = f"Failed to resolve generated mutation for {errorful_doc[errorful_idx]} to {correct_doc[correct_idx]} due to KeyError: {error}"
                    else:
                        msg = f"Failed to resolve generated mutation for {errorful_doc[errorful_idx]} to {correct_doc[correct_idx]} due to incorrect mutation\nResult of Mutation: {mutated_token}"
                    
                    if strict:
                        raise FailedToMeetStrictRequirementException(msg)
                    elif not silence_warnings:
                        print(msg)
                        print(f"Mutation: {mutation_labels}\nDefaulting to REPLACE (not preferred)")
                    cur_labels.append([f"<REPLACE param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"])
                else:
                    cur_labels.append(mutation_labels)
            elif process_lemma(correct_doc[correct_idx].lemma_) in token_lemmas: # Can copy
                param = -1
                for i, token in enumerate(errorful_doc):
                    if token.lemma_ == process_lemma(correct_doc[correct_idx].lemma_):
                        param = i
                        break
                if param == -1:
                    msg = f"COPY-REPLACE FAILED!\n\tThought {correct_doc[correct_idx].lemma_} was in {errorful_doc}.\n\tUsing normal replace (not preferred)"
                    if strict:
                        raise FailedToMeetStrictRequirementException(msg)
                    elif not silence_warnings:
                        print(msg)
                    cur_labels.append([f"<REPLACE param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"])
                else:
                    cur_labels.append([f"<COPY-REPLACE param=\"{param}\"/>"])
                    if not correct_doc[correct_idx].text in tokens: # Needs mutation
                        mutation_labels = link_with_mutation(errorful_doc[param], correct_doc[correct_idx])

                        # Verify mutation
                        failed_to_resolve = True
                        key_error = False
                        try:
                            mutated_token = verify_mutation(errorful_doc[param], [token for token in errorful_doc], errorful_idx, mutation_labels,
                                                            lemma_to_morph, nlp, correct_token=correct_doc[correct_idx], silence_warnings=silence_warnings)
                            if str(mutated_token) == correct_doc[correct_idx].text:
                                failed_to_resolve = False
                        except KeyError as e:
                            key_error = True
                            error = e

                        if failed_to_resolve:
                            if key_error:
                                msg = f"Failed to resolve generated mutation for {errorful_doc[param]} to {correct_doc[correct_idx]} due to KeyError: {error}"
                            else:
                                msg = f"Failed to resolve generated mutation for {errorful_doc[param]} to {correct_doc[correct_idx]} due to incorrect mutation\nResult of Mutation: {mutated_token}"
                            
                            if strict:
                                raise FailedToMeetStrictRequirementException(msg)
                            elif not silence_warnings:
                                print(msg)
                            cur_labels[-1] = [f"<REPLACE param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"]
                        else:
                            cur_labels[-1] += mutation_labels
            else:
                try:
                    cur_labels.append([f"<REPLACE param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"])
                except KeyError:
                    # Do operation in parts
                    tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
                    ids = tokenizer.encode(correct_doc[correct_idx].text)
                    word_labels = []
                    for i, id in enumerate(ids):
                        if i == len(ids) - 1:
                            word_labels.append(f"<REPLACE param=\"{id}\"/>")
                        else:
                            word_labels.append(f"<ADD param=\"{id}\"/>")
                    word_labels.reverse()
                    cur_labels.append(word_labels)
            labels.append(cur_labels)
            cur_labels = []
            op_idx += 1
            errorful_idx += 1
            correct_idx += 1
        elif operation == "ADD":
            if process_lemma(correct_doc[correct_idx].lemma_) in token_lemmas: # Can copy
                param = -1
                for i, token in enumerate(errorful_doc):
                    if process_lemma(token.lemma_) == process_lemma(correct_doc[correct_idx].lemma_):
                        param = i
                        break
                if param == -1:
                    msg = f"COPY-ADD FAILED!\n\tThought {correct_doc[correct_idx].lemma_} was in {errorful_doc}.\n\tUsing normal replace (not preferred)"
                    if strict:
                        raise FailedToMeetStrictRequirementException(msg)
                    elif not silence_warnings:
                        print(msg)
                    cur_labels.append([f"<ADD param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"])
                else:
                    cur_labels.append([f"<COPY-ADD param=\"{param}\"/>"])
                    if not correct_doc[correct_idx].text in tokens: # Needs mutation
                        mutation_labels = link_with_mutation(errorful_doc[param], correct_doc[correct_idx])

                        # Verify mutation
                        failed_to_resolve = True
                        key_error = False
                        try:
                            mutated_token = verify_mutation(errorful_doc[param], [token for token in errorful_doc], errorful_idx, mutation_labels,
                                                            lemma_to_morph, nlp, correct_token=correct_doc[correct_idx], silence_warnings=silence_warnings)
                            if str(mutated_token) == correct_doc[correct_idx].text:
                                failed_to_resolve = False
                        except KeyError as e:
                            key_error = True
                            error = e

                        if failed_to_resolve:
                            if key_error:
                                msg = f"Failed to resolve generated mutation for {errorful_doc[param]} to {correct_doc[correct_idx]} due to KeyError: {error}"
                            else:
                                msg = f"Failed to resolve generated mutation for {errorful_doc[param]} to {correct_doc[correct_idx]} due to incorrect mutation\nResult of Mutation: {mutated_token}"
                            
                            if strict:
                                raise FailedToMeetStrictRequirementException(msg)
                            elif not silence_warnings:
                                print(msg)
                            cur_labels[-1] = [f"<ADD param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"]
                        else:
                            cur_labels[-1] += mutation_labels
            else:
                try:
                    cur_labels.append([f"<ADD param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"])
                except KeyError:
                    # Do operation in parts
                    tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
                    ids = tokenizer.encode(correct_doc[correct_idx].text)
                    word_labels = []
                    for id in ids:
                        word_labels.append(f"<ADD param=\"{id}\"/>")
                    word_labels.reverse()
                    cur_labels.append(word_labels)
            op_idx += 1
            correct_idx += 1
     # Handle end of sequence
    if len(cur_labels) > 0:
        labels.append(cur_labels)
    else:
        labels.append([["<KEEP/>"]]) # For [EOS] token

    # Put modifications of the base token at the top of the token stack, more of an edge case
    # Basically handles the fact that decode logic needs modification of the base token first, then it can do adds
    # The minimum edit distance algorithm of course puts insertions first, then modifies the base token, so this just reverses that
    # In initial labeling, any modification of an ADD (only happens with COPY-ADD) is marked with -IN-PLACE which
    # sends it to the bottom of the stack here
    for i, cur_labels in enumerate(labels):
        new_cur_labels = []
        adds = []
        for cur_label in cur_labels:
            if not "ADD" in cur_label[0]:
                new_cur_labels.append(cur_label)
            else:
                adds.append(cur_label)
        adds.reverse()
        labels[i] = new_cur_labels + adds

    # Reduce dimensions
    for i, cur_labels in enumerate(labels):
        new_cur_labels = []
        for section in cur_labels:
            for label in section:
                new_cur_labels.append(label)
        labels[i] = new_cur_labels

    if verbose:
        print(f"Final labels: {labels}")
    labels = '\t'.join([' '.join(cur_labels) for cur_labels in labels])
    return labels

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

def main(input_file, output_file, lemma_to_morph, vocab, spacy_model="es_dep_news_trf", tokenizer_model="dccuchile/bert-base-spanish-wwm-cased",
         verify=False, verbose=False, silence_warnings=False, strict=False, n_cores=1):

    global nlp
    nlp = load_modified_nlp(model_path=spacy_model, tokenizer_path=tokenizer_model)
    vocab_index = create_vocab_index(vocab)

    start_time = time.time()
    failed_labels = 0
    with open(input_file, 'r') as f:
        lines = f.readlines()
        sentence_pairs = [(clean_text(lines[i]), clean_text(lines[i+1])) for i in range(0, len(lines), 3)]

        labels = []
        for i in range(0, len(sentence_pairs), n_cores):
            start_time = time.time()
            slice = sentence_pairs[i:min(i+n_cores, len(sentence_pairs))]
            new_labels = parallelize_function(slice, label_sentence_error_wrapper, n_cores, kwargs={"lemma_to_morph":lemma_to_morph, "vocab_index":vocab_index,
                                                                                                    "verbose":verbose, "silence_warnings":silence_warnings, "strict":strict})
            labels += new_labels
            failed_labels += len([label for label in new_labels if len(label) == 0])
            cur_time = time.time()
            print(f"===================================\nCompleted labeling for {len(new_labels)} sentences in {round(cur_time - start_time, 2)} seconds.\n{len([label for label in new_labels if len(label) == 0])} could not be labeled.\nAverage {round((cur_time - start_time) / n_cores, 3)} seconds per sentence")
            print(f"{round(min(i+n_cores, len(sentence_pairs)-1) / len(sentence_pairs) * 100, 1)}% done.")

    start_time = time.time()
    successful_labels = 0
    failed_verification = 0
    with open(output_file, 'w') as f:
        for i in range(0, len(sentence_pairs), n_cores):
            slice_sentences = sentence_pairs[i:min(i+n_cores, len(sentence_pairs))]
            slice_labels = labels[i:min(i+n_cores, len(sentence_pairs))]

            failed_this_round = 0
            if verify:
                print(f"===================================\nVerifying {len(slice_labels)} labels...")
                start_time = time.time()
                decoded_sentences = parallelize_function(zip(list(zip(*slice_sentences))[0], slice_labels), apply_labels_error_wrapper, n_cores, kwargs={"lemma_to_morph":lemma_to_morph, "vocab":vocab,
                                                                                                                                                   "verbose":verbose, "silence_warnings":silence_warnings})
            for j in range(len(slice_sentences)):
                if verify and slice_sentences[j][1] != decoded_sentences[j]:
                    if slice_labels[j] != "" and decoded_sentences[j] != "": # No need to warn if this sentence already failed labeling or errored during apply_labels
                        if not silence_warnings:
                            print(f"VERIFY FAILED!\nReport:\n\tErrorful Sentence:{slice_sentences[j][0]}\n\tGenerated Labels:{slice_labels[j]}\n\tTarget:{slice_sentences[j][1]}\n\tResult from Decode:{decoded_sentences[j]}")
                        failed_this_round += 1
                        failed_verification += 1
                    continue
                else:
                    f.write(f"{slice_sentences[j][0]}\n{slice_labels[j]}\n{slice_sentences[j][1]}\n\n")
                    successful_labels += 1
            if verify:
                cur_time = time.time()
                print(f"Completed verification for {len(decoded_sentences)} sentences in {round(cur_time - start_time, 2)} seconds.\n{failed_this_round} sentences failed verification.\nAverage {round((cur_time - start_time) / n_cores, 3)} seconds per sentence")
                print(f"{round(min(i+n_cores, len(sentence_pairs)-1) / len(sentence_pairs) * 100, 1)}% done.")
            
    print(f"===============================\nFailed to label {failed_labels + failed_verification} sentences.")
    print(f"Successfully labeled {successful_labels} sentences")

    new_dict_path = "lang_def/morpho_dict_updated.json"
    if verbose:
        print(f"Exporting updated morphology dictionary to {new_dict_path}")
    with open(new_dict_path, 'w') as f:
        json.dump(lemma_to_morph, f)

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to a file with sentences in Spanish on each line")
    parser.add_argument("output_file", help="output path")
    parser.add_argument("--dict_file", default="lang_def/morpho_dict_updated.json", help="path to the dictionary file which supplies different morphological forms for a word")
    parser.add_argument("--vocab_file", default="lang_def/vocab.txt", help="path to the vocab file containing all words in your model's vocabulary")
    parser.add_argument("--spacy_model", default="es_dep_news_trf", help="spaCy model to use")
    parser.add_argument("--tokenizer_model", default="dccuchile/bert-base-spanish-wwm-cased", help="Tokenizer model to use (local or HuggingFace path)")

    parser.add_argument("--n_cores", default=1, type=int, help="Number of cores to use (1 for no multi-processing)")

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

    if not args.output_file:
        input_path = Path(args.input_file)
        output_file = input_path.parent / f"{input_path.stem}_labeled{input_path.suffix}"
    else:
        output_file = args.output_file

    main(args.input_file, output_file, load_morpho_dict(args.dict_file), load_vocab(args.vocab_file), spacy_model=args.spacy_model, tokenizer_model=args.tokenizer_model,
         verify=args.verify, verbose=args.verbose, silence_warnings=args.silence_warnings, strict=args.strict, n_cores=args.n_cores)