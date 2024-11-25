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
import argparse
from pathlib import Path
import numpy as np
import spacy
from bs4 import BeautifulSoup
from unidecode import unidecode

from utils.utils import load_morpho_dict, load_vocab, get_path, create_vocab_index, apply_labels, mutate, CONTEXT_WINDOW

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
        else:
            if errorful_sequence[i] != correct_sequence[i]:
                final_sequence.append(correct_sequence[i])

    if correct_token.text[0].isupper(): # Check if capitalization needs to be applied
        final_sequence.append("<MUTATE param=\"CAPITALIZE-TRUE\"/>")
    elif correct_token.text[0].islower() and not errorful_token.text[0].islower(): # Check if capitalization needs to be removed
        final_sequence.append("<MUTATE param=\"CAPITALIZE-FALSE\"/>")

    return final_sequence

def verify_mutation(token, sentence, token_idx, labels, lemma_to_morph, nlp):
    token = [token]
    for label in labels:
        if "MUTATE" in label:
            left_context = []
            for segment in sentence[max(0, token_idx - CONTEXT_WINDOW):token_idx]:
                if not isinstance(segment, list):
                    segment = [segment]
                for cur_token in segment:
                    left_context.append(cur_token)
            right_context = []
            for segment in sentence[token_idx+1:min(len(sentence), token_idx + CONTEXT_WINDOW + 1)]:
                if not isinstance(segment, list):
                    segment = [segment]
                for cur_token in segment:
                    right_context.append(cur_token)
            with_context = left_context + token + right_context
            all_strs = [str(token) for token in with_context]
            new_segment_with_context = [token for token in nlp(' '.join(all_strs))]
            token = new_segment_with_context[len(left_context):len(token) + len(left_context)]
            if not isinstance(token, list):
                token = [token]

            param = BeautifulSoup(label, features="html.parser").find_all(True)[0].get("param", "").upper()
            token = [mutate(token[0], param, lemma_to_morph, nlp)]
    return token[0]

def label_sentence(errorful, correct, lemma_to_morph, vocab_index, nlp, verbose=False):

    errorful_doc = nlp(errorful)
    correct_doc = nlp(correct)

    if verbose:
        print(f"=================================\nTesting labels for:\n{errorful}\nto\n{correct}")

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
    token_lemmas = {unidecode(token.lemma_) for token in errorful_doc}
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
            if unidecode(errorful_doc[errorful_idx].lemma_) == unidecode(correct_doc[correct_idx].lemma_): # Can mutate
                mutation_labels = link_with_mutation(errorful_doc[errorful_idx], correct_doc[correct_idx])
                
                # Verify mutation
                failed_to_resolve = True
                key_error = False
                try:
                    mutated_token = verify_mutation(errorful_doc[errorful_idx], [token for token in errorful_doc], errorful_idx, mutation_labels, lemma_to_morph, nlp)
                    if str(mutated_token) == correct_doc[correct_idx].text:
                        failed_to_resolve = False
                except KeyError as e:
                    key_error = True
                    error = e

                if failed_to_resolve:
                    if key_error:
                        print(f"Failed to resolve generated mutation for {errorful_doc[errorful_idx]} to {correct_doc[correct_idx]} due to KeyError: {error}")
                    else:
                        print(f"Failed to resolve generated mutation for {errorful_doc[errorful_idx]} to {correct_doc[correct_idx]} due to incorrect mutation\nResult of Mutation: {mutated_token}")
                    print(f"Mutation: {mutation_labels}\nDefaulting to REPLACE (not preferred)")
                    cur_labels.append([f"<REPLACE param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"])
                else:
                    cur_labels.append(mutation_labels)
            elif unidecode(correct_doc[correct_idx].lemma_) in token_lemmas: # Can copy
                param = -1
                for i, token in enumerate(errorful_doc):
                    if token.lemma_ == unidecode(correct_doc[correct_idx].lemma_):
                        param = i
                        break
                if param == -1:
                    print(f"COPY-REPLACE FAILED!\n\tThought {correct_doc[correct_idx].lemma_} was in {errorful_doc}.\n\tUsing normal replace (not preferred)")
                    cur_labels.append([f"<REPLACE param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"])
                else:
                    cur_labels.append([f"<COPY-REPLACE param=\"{param}\"/>"])
                    if not correct_doc[correct_idx].text in tokens: # Needs mutation
                        mutation_labels = link_with_mutation(errorful_doc[param], correct_doc[correct_idx])

                        # Verify mutation
                        failed_to_resolve = True
                        key_error = False
                        try:
                            mutated_token = verify_mutation(errorful_doc[param], [token for token in errorful_doc], errorful_idx, mutation_labels, lemma_to_morph, nlp)
                            if str(mutated_token) == correct_doc[correct_idx].text:
                                failed_to_resolve = False
                        except KeyError as e:
                            key_error = True
                            error = e

                        if failed_to_resolve:
                            if key_error:
                                print(f"Failed to resolve generated mutation for {errorful_doc[param]} to {correct_doc[correct_idx]} due to KeyError: {error}")
                            else:
                                print(f"Failed to resolve generated mutation for {errorful_doc[param]} to {correct_doc[correct_idx]} due to incorrect mutation\nResult of Mutation: {mutated_token}")
                            print(f"Mutation: {mutation_labels}\nDefaulting to REPLACE (not preferred)")
                            cur_labels[-1] = [f"<REPLACE param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"]
                        else:
                            cur_labels[-1] += mutation_labels
            else:
                cur_labels.append([f"<REPLACE param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"])
            labels.append(cur_labels)
            cur_labels = []
            op_idx += 1
            errorful_idx += 1
            correct_idx += 1
        elif operation == "ADD":
            if unidecode(correct_doc[correct_idx].lemma_) in token_lemmas: # Can copy
                param = -1
                for i, token in enumerate(errorful_doc):
                    if unidecode(token.lemma_) == unidecode(correct_doc[correct_idx].lemma_):
                        param = i
                        break
                if param == -1:
                    print(f"COPY-ADD FAILED!\n\tThought {correct_doc[correct_idx].lemma_} was in {errorful_doc}.\n\tUsing normal replace (not preferred)")
                    cur_labels.append([f"<ADD param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"])
                else:
                    cur_labels.append([f"<COPY-ADD param=\"{param}\"/>"])
                    if not correct_doc[correct_idx].text in tokens: # Needs mutation
                        mutation_labels = link_with_mutation(errorful_doc[param], correct_doc[correct_idx])

                        # Verify mutation
                        failed_to_resolve = True
                        key_error = False
                        try:
                            mutated_token = verify_mutation(errorful_doc[param], [token for token in errorful_doc], errorful_idx, mutation_labels, lemma_to_morph, nlp)
                            if str(mutated_token) == correct_doc[correct_idx].text:
                                failed_to_resolve = False
                        except KeyError as e:
                            key_error = True
                            error = e

                        if failed_to_resolve:
                            if key_error:
                                print(f"Failed to resolve generated mutation for {errorful_doc[param]} to {correct_doc[correct_idx]} due to KeyError: {error}")
                            else:
                                print(f"Failed to resolve generated mutation for {errorful_doc[param]} to {correct_doc[correct_idx]} due to incorrect mutation\nResult of Mutation: {mutated_token}")
                            print(f"Mutation: {mutation_labels}\nDefaulting to ADD (not preferred)")
                            cur_labels[-1] = [f"<ADD param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"]
                        else:
                            cur_labels[-1] += mutation_labels
            else:
                cur_labels.append([f"<ADD param=\"{vocab_index[correct_doc[correct_idx].text]}\"/>"])
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

def main(input_file, output_file, lemma_to_morph, vocab, verify=False, verbose=False):

    nlp = spacy.load("es_dep_news_trf")
    vocab_index = create_vocab_index(vocab)

    failed_labels = 0
    with open(input_file, 'r') as f:
        lines = f.readlines()
        sentence_pairs = [(lines[i], lines[i+1]) for i in range(0, len(lines), 3)]

        labels = []
        for sentences in sentence_pairs:
            try:
                cur_labels = label_sentence(sentences[0], sentences[1], lemma_to_morph, vocab_index, nlp, verbose=verbose)
                labels.append(cur_labels)
            except Exception as e:
                print(f"Failed to generate labels due to {e}\n\tErrorful sentence: {sentences[0]}\n\tCorrect sentence: {sentences[1]}")
                failed_labels += 1

    successful_labels = 0
    with open(output_file, 'w') as f:
        for i in range(len(sentence_pairs)):
            errorful_sentence = sentence_pairs[i][0]
            correct_sentence = sentence_pairs[i][1]
            token_labels = labels[i]

            if verify:
                for i in range(len(sentence_pairs)):
                    errorful_sentence = sentence_pairs[i][0]
                    correct_sentence = sentence_pairs[i][1]
                    token_labels = labels[i]

                    decoded_sentence = apply_labels(nlp(errorful_sentence), token_labels.split('\t'), lemma_to_morph, vocab, nlp)
                    if decoded_sentence != correct_sentence:
                        print(f"VERIFY FAILED!\nReport:\n\tErrorful Sentence:{errorful_sentence}\n\tGenerated Labels:{token_labels}\n\tTarget:{correct_sentence}\n\tResult from Decode:{decoded_sentence}")
                        failed_labels += 1
                        continue
            
            f.write(f"{errorful_sentence}\n{token_labels}\n{correct_sentence}\n\n")
            successful_labels += 1
    print(f"Failed to label {failed_labels} sentences.")
    print(f"Successfully labeled {successful_labels} sentences")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to a file with sentences in Spanish on each line")
    parser.add_argument("output_file", help="output path")
    parser.add_argument("--dict_file", default="lang_def/morpho_dict.json", help="path to the dictionary file which supplies different morphological forms for a word")
    parser.add_argument("--vocab_file", default="lang_def/vocab.txt", help="path to the vocab file containing all words in your model's vocabulary")
    parser.add_argument("--verify",
                        help="generated token labels will be verified as correct by running the decode algorithm",
                        action="store_true")
    parser.add_argument("-v", "--verbose",
                        help="Print debugging statements",
                        action="store_true")

    args = parser.parse_args()

    if not args.output_file:
        input_path = Path(args.input_file)
        output_file = input_path.parent / f"{input_path.stem}_labeled{input_path.suffix}"
    else:
        output_file = args.output_file

    main(args.input_file, output_file, load_morpho_dict(args.dict_file), load_vocab(args.vocab_file), verify=args.verify, verbose=args.verbose)