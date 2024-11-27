'''
File: decode.py
Author: Amber Converse
Purpose: This script takes errorful sentences + token-level labels and decodes them into a corrected sentence.
'''

import argparse
from pathlib import Path

from utils.utils import apply_labels, load_modified_nlp, load_morpho_dict, load_vocab, clean_text

def correct_errorful_sentences(input_file, output_file, dict_file, vocab_file, silence_warnings=False):

    nlp = load_modified_nlp()
    vocab = load_vocab(vocab_file)
    lemma_to_morph = load_morpho_dict(dict_file)

    corrected_sentences = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        pairs = [(clean_text(lines[i]), clean_text(lines[i+1])) for i in range(0, len(lines), 3)]

        for sentence_label in pairs:
            try:
                corrected_sentence = apply_labels(nlp(sentence_label[0]), sentence_label[1].split('\t'), lemma_to_morph, vocab, nlp)
                corrected_sentences.append(corrected_sentence)
            except KeyError as e:
                if not silence_warnings:
                    print(f"Failed to apply labels due to KeyError {e}.\nSentence: {sentence_label[0]}\nLabels: {sentence_label[1]}")

    with open(output_file, 'w') as f:
        for corrected_sentence in corrected_sentences:
            f.write(f"{corrected_sentence}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to a file with a sentence on one line, the respective token-level labels on the next line, and a blank line before the next sentence")
    parser.add_argument("output_file", help="optional, defines the output path to place the parsed sentences in")
    parser.add_argument("--dict_file", default="lang_def/morpho_dict_updated.json", help="path to the dictionary file which supplies different morphological forms for a word")
    parser.add_argument("--vocab_file", default="lang_def/vocab.txt", help="path to the vocab file containing all words in your model's vocabulary")
    parser.add_argument("-sw", "--silence_warnings",
                        help="Silence warnings",
                        action="store_true")

    args = parser.parse_args()

    input_file = args.input_file
    if not args.output_file:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_parsed{input_path.suffix}"
    else:
        output_file = args.output_file

    correct_errorful_sentences(input_file, output_file, args.dict_file, args.vocab_file, silence_warnings=args.silence_warnings)