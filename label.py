'''
File: label.py
Author: Amber Converse
Purpose: This script takes errorful sentences + target sentences and translates them into token-level edits
    using shortest edit distance.
'''

import argparse
from pathlib import Path

def label_sentences(input_file, output_file, dict_file, vocab_file, token_type=[]):
    '''
    Stub
    '''
    raise NotImplementedError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to a file with an errorful sentence on one line, the target sentence on the next, and a blank line before the next sentence")
    parser.add_argument("output_file", help="optional, defines the output path to place the parsed sentences in. If no output path is supplied, it is placed in the same directory as the input file w/ the name [input file name]_labeled.txt")
    parser.add_argument("-d", "--dict-file", help="path to the dictionary file which supplies different morphological forms for a word")
    parser.add_argument("-v", "--vocab-file", help="path to the vocab file containing all words in your model's vocabulary")
    parser.add_argument("-t", "--token",
                        help="followed by the move/replace types to be used for token-level labels",
                        nargs="*")

    args = parser.parse_args()

    input_file = args.input_file
    if not args.output_file:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_parsed{input_path.suffix}"
    else:
        output_file = args.output_file

    label_sentences(input_file, output_file, args.dict_file, args.vocab_file, args.token if args.token else [])