'''
File: test_decode.py
Author: Amber Converse
Purpose: Unit tests for decode.py and related functions
'''
import sys
sys.path.append("..")

import unittest
import spacy

from utils.utils import load_vocab, load_morpho_dict, apply_labels

BASE_TESTS = [("Voy al tienda.",
               "Voy a la tienda.",
               "<KEEP/>\t<REPLACE param=\"1013\"/>\t<ADD param=\"1030\"/>\t<KEEP/>\t<KEEP/>"
               ),
              ("espero que estudias más.",
               "Espero que estudies más.",
               "<MUTATE param=\"CAPITALIZE-TRUE\"/>\t<KEEP/>\t<MUTATE param=\"MOOD-SUB\"/>\t<KEEP/>\t<KEEP/>\t<KEEP/>"
               ),
              ("Soy mudando a Chicago",
               "Estoy mudando a Chicago.",
               "<REPLACE param=\"2040\"/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<ADD param=\"1009\"/>"
               ),
              ("Soy mudar Chicago",
               "Estoy mudando a Chicago.",
               "<REPLACE param=\"2040\"/>\t<MUTATE param=\"MOOD-GER\"/>\t<ADD param=\"1013\"/>\t<ADD param=\"1009\"/>"
               ),
               ("Yo gusto desarrollar progamas.",
               "Me gusta desarrollar progamas.",
               "<MUTATE param=\"CASE-DAT\"/> <MUTATE param=\"PRONOUN_TYPE-CLITIC\"/> <MUTATE param=\"CAPITALIZE-TRUE\"/>\t<MUTATE param=\"PERSON-3\"/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>"
               ),
               ("Estoy escribieron un programa.",
               "Estoy escribiendo un programa",
               "<KEEP/>\t<MUTATE param=\"POS-VERB\"/> <MUTATE param=\"MOOD-GER\"/>\t<KEEP/>\t<KEEP/>\t<DELETE/>\t<KEEP/>"
               )]

COPY_TESTS = [("Almorzar quiero.",
               "Quiero almorzar.",
               "<COPY-REPLACE param=\"1\"/> <MUTATE param=\"CAPITALIZE-TRUE\"/>\t<COPY-REPLACE param=\"0\"/> <MUTATE param=\"CAPITALIZE-FALSE\"/>\t<KEEP/>\t<KEEP/>"
               ),
              ("yo ellos faltamos dinero, para eso nosotros busque Trabajo",
               "Nosotros faltamos dinero, por eso nosotros buscamos trabajo.",
               "<DELETE/>\t<COPY-REPLACE param=\"0\"/> <MUTATE param=\"NUMBER-PLUR\"/> <MUTATE param=\"CAPITALIZE-TRUE\"/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<REPLACE param=\"1096\"/>\t<KEEP/>\t<KEEP/>\t<MUTATE param=\"NUMBER-PLUR\"/> <MUTATE param=\"MOOD-IND\"/> <MUTATE param=\"PERSON-1\"/>\t<MUTATE param=\"CAPITALIZE-FALSE\"/>\t<ADD param=\"1009\"/>"),
              ("haciendo qué nosotros estuve",
               "¿Qué estábamos haciendo nosotros?",
               "<REPLACE param=\"1067\"/>\t<MUTATE param=\"CAPITALIZE-TRUE\"/>\t<COPY-ADD param=\"0\"/> <COPY-ADD param=\"3\"/> <MUTATE param=\"NUMBER-PLUR\"/> <MUTATE param=\"TIME-IMP\"/>\t<REPLACE param=\"1064\"/>\t<KEEP/>"),
              ("soy in la parque y pronto voy a estar en Chicago.",
               "Estoy en el parque y pronto voy a estar en Chicago.",
               "<COPY-REPLACE param=\"8\"/> <MUTATE param=\"NUMBER-SING\"/> <MUTATE param=\"MOOD-IND\"/> <MUTATE param=\"TIME-PRES\"/> <MUTATE param=\"PERSON-1\"/> <MUTATE param=\"CAPITALIZE-TRUE\"/>\t<COPY-REPLACE param=\"9\"/>\t<MUTATE param=\"GENDER-MASC\"/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>"
               ),
               # Not a grammar error, just needed to test REPLACE -> ADD-COPY + MUTATE in same token
               ("queremos que nosotros vamos al parque.",
               "Yo quiero que nosotros vamos al parque.",
               "<COPY-REPLACE param=\"2\"/> <MUTATE param=\"NUMBER-SING\"/> <MUTATE param=\"CAPITALIZE-TRUE\"/>\t<COPY-ADD param=\"0\"/> <MUTATE param=\"NUMBER-SING\"/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>\t<KEEP/>"
               )]


class TestApply(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("es_dep_news_trf")
        self.vocab = load_vocab("../lang_def/vocab.txt")
        self.lemma_to_morph = load_morpho_dict("../lang_def/morpho_dict.json")

    def test_decode(self):
        for TEST in BASE_TESTS:
            self.assertEqual(apply_labels(self.nlp(TEST[0]),
                                          TEST[2].split('\t'),
                                          self.lemma_to_morph,
                                          self.vocab,
                                          self.nlp),
                            TEST[1])
            
    def test_copy(self):
        for TEST in COPY_TESTS:
            self.assertEqual(apply_labels(self.nlp(TEST[0]),
                                          TEST[2].split('\t'),
                                          self.lemma_to_morph,
                                          self.vocab,
                                          self.nlp),
                            TEST[1])

if __name__ == "__main__":
    unittest.main()