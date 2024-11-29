'''
File: decode.py
Author: Amber Converse
Purpose: This script takes errorful sentences + token-level labels and decodes them into a corrected sentence.
'''

import argparse
import traceback
import warnings
from pathlib import Path

from utils.utils import apply_labels, load_modified_nlp, load_morpho_dict, load_vocab, clean_text, parallelize_function

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

def correct_errorful_sentences(input_file, output_file, dict_file, vocab_file,
                               spacy_model="es_dep_news_trf", tokenizer_model="dccuchile/bert-base-spanish-wwm-cased", verbose=False, silence_warnings=False, n_cores=1):

    global nlp
    nlp = load_modified_nlp(model_path=spacy_model, tokenizer_path=tokenizer_model)
    vocab = load_vocab(vocab_file)
    lemma_to_morph = load_morpho_dict(dict_file)

    corrected_sentences = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        pairs = [(clean_text(lines[i]), clean_text(lines[i+1])) for i in range(0, len(lines), 3)]

        for i in range(0, len(pairs), n_cores):
            slice = pairs[i:i:min(i+n_cores, len(pairs)-1)]

            candidate_sentences = parallelize_function(slice, apply_labels_error_wrapper, n_cores, kwargs={"lemma_to_morph":lemma_to_morph, "vocab":vocab,
                                                                                                           "verbose":verbose, "silence_warnings":silence_warnings})
            corrected_sentences += [sentence for sentence in candidate_sentences if sentence != ""]

    with open(output_file, 'w') as f:
        for corrected_sentence in corrected_sentences:
            f.write(f"{corrected_sentence}\n\n")

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to a file with a sentence on one line, the respective token-level labels on the next line, and a blank line before the next sentence")
    parser.add_argument("output_file", help="optional, defines the output path to place the parsed sentences in")
    parser.add_argument("--dict_file", default="lang_def/morpho_dict_updated.json", help="path to the dictionary file which supplies different morphological forms for a word")
    parser.add_argument("--vocab_file", default="lang_def/vocab.txt", help="path to the vocab file containing all words in your model's vocabulary")
    parser.add_argument("--spacy_model", default="es_dep_news_trf", help="spaCy model to use")
    parser.add_argument("--tokenizer_model", default="dccuchile/bert-base-spanish-wwm-cased", help="Tokenizer model to use (local or HuggingFace path)")

    parser.add_argument("--n_cores", default=1, type=int, help="Number of cores to use (1 for no multi-processing)")

    parser.add_argument("-v", "--verbose",
                        help="Print debugging statements",
                        action="store_true")
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

    correct_errorful_sentences(input_file, output_file, args.dict_file, args.vocab_file, spacy_model=args.spacy_model, tokenizer_model=args.tokenizer_model,
                               verbose=args.verbose, silence_warnings=args.silence_warnings, n_cores=args.n_cores)