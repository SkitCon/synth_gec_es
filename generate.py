'''
File: generate.py
Author: Amber Converse
Purpose: This script generates synthetic errorful sentences from well-formed Spanish sentences in a corpus.
'''

import argparse
from pathlib import Path

def generate_errorful_sentences(input_file, output_file, num_sentences, seq2seq=False, token_types=[]):
    '''
    Stub
    '''
    raise NotImplementedError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to a file with sentences in Spanish on each line")
    parser.add_argument("output_file", help="output path")
    parser.add_argument("-n", "--num-sentences", help="the number of errorful sentences that will be generated from each correct sentence in the supplied corpus. By default 1")
    parser.add_argument("-s", "--seq2seq",
                        help="the output synthetic data will include the raw errorful sentence unlabeled for use in a traditional NMT-based seq2seq GEC system (as with BART or T5)",
                        action="store_true")
    parser.add_argument("-t", "--token",
                        help="followed by the move/replace types to be used for token-level labels",
                        nargs="*")

    args = parser.parse_args()

    input_file = args.input_file
    if not args.output_file:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_synth{input_path.suffix}"
    else:
        output_file = args.output_file
    
    if not args.num_sentences:
        num_sentences = 1
    else:
        num_sentences = args.num_sentences

    generate_errorful_sentences(input_file, output_file, num_sentences, args.seq2seq, args.token if args.token else [])