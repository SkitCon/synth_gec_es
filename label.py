'''
File: label.py
Author: Amber Converse
Purpose: This script takes errorful sentences + target sentences and translates them into token-level edits
    using shortest edit distance.
'''

import argparse
from pathlib import Path
import numpy as np
import spacy

COST = {"KEEP": 0,
        "DELETE": 0,
        "PRE-ADD": 2,
        "POST-ADD": 2,
        "PRE-COPY": 1,
        "POST-COPY": 1,
        "MUTATE": 0,
        "REPLACE": 3}

def traceback(dp, edits):
    '''
    Stub
    '''
    raise NotImplementedError

def create_labels(errorful_sentence, correct_sentence)
    '''
    Assumes replace and relative indexing, WIP
    '''
    dp = np.zeros((len(errorful_sentence)+1, len(correct_sentence)+1))
    dp[0, :] = list(range(dp.shape[1])) * COST["PRE-ADD"]
    dp[:, 0] = list(range(dp.shape[0])) * COST["DELETE"]
    edits = np.ndarray([[""] * len(correct_sentence)+1] * len(errorful_sentence)+1)

    edits[0, :] = ["PRE-ADD"] * edits.shape[1]
    edits[:, 0] = ["DELETE"] * edits.shape[0]

    for i in range(1, np.shape[1])+1:
        for j in range(np.shape[0]+1):

            # KEEP
            if errorful_sentence[i] == correct_sentence[j]:
                dp[i, j] = dp[i-1, j-1] + COST["KEEP"]
                edits[i, j] = "KEEP"

            # TODO: DELETE

            # TODO: PRE-ADD and POST-ADD

            # TODO: MUTATE

            # TODO: REPLACE
            
            # TODO
            # Find way to map moves to dp matrix
    
    return traceback(dp, edits)


def label_sentences(input_file, output_file, dict_file, vocab_file):
    '''
    This function takes errorful sentences + target sentences and translates them into token-level edits
    using shortest edit distance.

    :param input_file (str): path to the input file
    :param output_file (str): path to the output file
    :param dict_file (str): path to the dict_file
    :param vocab_file (str): path to the vocab_file
    :token_type ([str]): the move/replace types to use
    '''
    nlp = spacy.load("es_dep_news_trf")

    errorful_sentences = []
    token_labels = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            errorful_sentence = [token.text for token in nlp(lines[i])]
            correct_sentence = [token.text for token in nlp(lines[i+1])]
            cur_token_labels = create_labels(errorful_sentence, correct_sentence)
            errorful_sentences.append(errorful_sentence)
            token_labels.append(cur_token_labels)
    
    # Output errorful sentences and token_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to a file with an errorful sentence on one line, the target sentence on the next, and a blank line before the next sentence")
    parser.add_argument("output_file", help="optional, defines the output path to place the parsed sentences in. If no output path is supplied, it is placed in the same directory as the input file w/ the name [input file name]_labeled.txt")
    parser.add_argument("-d", "--dict-file", help="path to the dictionary file which supplies different morphological forms for a word")
    parser.add_argument("-v", "--vocab-file", help="path to the vocab file containing all words in your model's vocabulary")

    args = parser.parse_args()

    input_file = args.input_file
    if not args.output_file:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_parsed{input_path.suffix}"
    else:
        output_file = args.output_file

    label_sentences(input_file, output_file, args.dict_file, args.vocab_file)