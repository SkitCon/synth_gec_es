'''
File: test_decode.py
Author: Amber Converse
Purpose: Unit tests for decode.py and related functions
'''
import sys
sys.path.append("..")

import unittest
import spacy

from utils.utils import load_vocab, load_lex_dict, apply_labels

TESTS = [("Voy al tienda.", "<KEEP/>\t<REPLACE param=\"1013\"/>\t<PRE-ADD param=\"1030\"/>\t<KEEP/>")]


class TestApply(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("es_dep_news_trf")
        self.vocab = load_vocab("lang_def/vocab.txt")
        self.lemma_to_morph = load_lex_dict("lang_def/morph_dict.json")

    def test_replace(self):
        self.assertEqual(apply_labels(self.nlp(TESTS[0][0]),
                                               TESTS[0][1].split('\t'),
                                               self.lemma_to_morph,
                                               self.vocab,
                                               self.nlp),
                         "Voy a la tienda.")

if __name__ == "__main__":
    unittest.main()