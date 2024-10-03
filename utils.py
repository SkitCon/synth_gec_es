'''
File: utils.py
Author: Amber Converse
Purpose: Contains a variety of functions for use in generating synthetic errorful sentences in
    Spanish
'''
from custom_errors import InvalidLabelException

GENDERS = ["MASC", "FEM"] # No non-binary :( RAE dice que no
NUMBERS = ["SING", "PLU"]
MOODS = ["IND", "POS-IMP", "NEG-IMP", "SUBJ", "PROG", "PERF", "PERF-SUBJ", "GER", "PAST-PART", "INF"]
TIMES = ["PRES", "PRET", "IMPERF", "COND", "FUT"]
PERSONS = ["1", "2", "3"]
HIERARCHY_DEF = {"NUMBER": 0,
                    "GENDER": 1,
                    "MOOD": 1,
                        "TIME": 2,
                            "PERSON": 3}

def load_lex_dict(filename):
    '''
    Loads the lexical dictionary in filename
    
    :param filename (str): the path to the dictionary file

    :return lemma_to_lex (dict): a dictionary that maps the base form of a word to all of its
        lexical forms in hierarhical format of
            LEMMA
            |- NUMBER
                |- GENDER,
                |- MOOD,
                    |- TIME
                        |- PERSON
    :return lex_to_lemma (dict): a dictionary with all lexical forms as keys and the lemma as
        values. Essentially, a lemmatizer as dict
    '''
    lemma_to_lex = {}
    lex_to_lemma = {}
    with open(filename, 'r') as f:
        pass
    return lemma_to_lex, lex_to_lemma

def mutate(word, label_param, lemma_to_lex, lemmatizer):
    '''
    Applies mutate to the given word

    :param word (str): the word for mutate to applied to
    :param label_param (int|str): the parameter for the label (if applicable)
    :param lemma_to_lex (dict): a dictionary that maps the base form of a word to all of its
        lexical forms in hierarhical format of
            LEMMA
            |- NUMBER
                |- GENDER,
                |- MOOD,
                    |- TIME
                        |- PERSON
    :param lemmatizer (func(str)): a function that takes a word as input and returns the lemma

    :return corrected_word (str): the mutated word
    '''
    if label_param.startswith("CAPITALIZE"):
            if label_param.split("-")[1] == "TRUE":
                return word[0].upper() + word[1:]
            else:
                return word[0].lower() + word[1:]
    lemma = lemmatizer(word)
    lexemes = lemma_to_lex[word]
    cur_path = []
    for number in NUMBERS:
        for gender in GENDERS:
            if lexemes[number][gender] == word:
                cur_path = [number, gender]
                break
        if cur_path:
            break
        for mood in MOODS:
            for time in TIMES:
                for person in PERSONS:
                    if lexemes[number][mood][time][person] == word:
                        cur_path == [number, mood, time, person]
                        break
                if cur_path:
                    break
            if cur_path:
                break
        if cur_path:
            break
    
    new_path = cur_path.copy()
    new_path[HIERARCHY_DEF[label_param.split('-')[0]]] = label_param.split('-')[1]

    # TODO return simple case
    # TODO return complex case (prog, perf)

    return mutated_word


def apply_label(word, label, label_param, lemma_to_lex, lemmatizer, vocab):
    '''
    Apply label to a word. Cannot be any form of MOVE.

    :param word (str): the word for label to applied to
    :param label (str): the label to be applied to word
    :param label_param (int|str): the parameter for the label (if applicable)
    :param lemma_to_lex (dict): a dictionary that maps the base form of a word to all of its
        lexical forms in hierarhical format of
            LEMMA
            |- NUMBER
                |- GENDER,
                |- MOOD,
                    |- TIME
                        |- PERSON
    :param lemmatizer (func(str)): a function that takes a word as input and returns the lemma
    :param vocab ([str]): a list of all words in dictionary, aligned with model vocabulary

    :return corrected_word (str): the corrected word based on label
    '''
    if label not in ["KEEP", "PRE-ADD", "POST-ADD", "MUTATE", "REPLACE"]:
        raise InvalidLabelException()
    
    if label == "KEEP":
        return word
    elif label == "PRE-ADD":
        return vocab[label_param] + word
    elif label == "POST-ADD":
        return word + vocab[label_param]
    elif label == "MUTATE":
        return mutate(word, label_param, lemma_to_lex, lemmatizer)
    elif label == "REPLACE":
        return vocab[label_param]
        



def apply_labels(sentence, labels, lemma_to_lex, lemmatizer, vocab):
    '''
    Applies the token-level GEC labels to the given sentence.

    sentence and labels must be the same length.

    :param sentence ([str]): tokenized errorful sentence
    :param labels ([str]): list of token-level labels to be applied to errorful sentence for correction
    :param lemma_to_lex (dict): a dictionary that maps the base form of a word to all of its
        lexical forms in hierarhical format of
            LEMMA
            |- NUMBER
                |- GENDER,
                |- MOOD,
                    |- TIME
                        |- PERSON
    :param lemmatizer (func(str)): a function that takes a word as input and returns the lemma
    :param vocab ([str]): a list of all words in dictionary, aligned with model vocabulary

    :return corrected sentence (str): the corrected sentence
    '''
    corrected_sentence = []
    for i, label in enumerate(labels):
        pass
        # TODO base case
        # TODO move case
