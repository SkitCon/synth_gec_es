'''
File: decode.py
Author: Amber Converse
Purpose: This script takes errorful sentences + token-level labels and decodes them into a corrected sentence.
'''

import argparse
from pathlib import Path

def correct_errorful_sentences(input_file, output_file, dict_file, vocab_file):
    '''
    Stub
    '''
    raise NotImplementedError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to a file with a sentence on one line, the respective token-level labels on the next line, and a blank line before the next sentence")
    parser.add_argument("output_file", help="optional, defines the output path to place the parsed sentences in")
    parser.add_argument("-d", "--dict-file", help="path to the dictionary file which supplies different morphological forms for a word")
    parser.add_argument("-v", "--vocab-file", help="path to the vocab file containing all words in your model's vocabulary")

    args = parser.parse_args()

    input_file = args.input_file
    if not args.output_file:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_parsed{input_path.suffix}"
    else:
        output_file = args.output_file

    correct_errorful_sentences(input_file, output_file, args.dict_file, args.vocab_file)