'''
File: utils.py
Author: Amber Converse
Purpose: Contains a variety of functions for use in generating synthetic errorful sentences in
    Spanish
'''
from bs4 import BeautifulSoup
from collections import deque
from custom_errors import InvalidLabelException
from custom_errors import NotInDictionaryException

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
        **IMPORTANT**: Note that in this dictionary the infinitive is given a number, time, and
            person (sing, pres, 1st) to enable transformations FROM infinitives. Essentially, this
            determination defines the *default* form of a verb which previously did not have these
            characteristics.
    :return lex_to_lemma (dict): a dictionary with all lexical forms as keys and the lemma as
        values. Essentially, a lemmatizer as dict
    '''
    lemma_to_lex = {}
    lex_to_lemma = {}
    with open(filename, 'r') as f:
        pass
    return lemma_to_lex, lex_to_lemma

def follow_path(dictionary, path):
    '''
    Simple helper function for a hierarchical dict. Recursively follow the given path into the
    dictionary

    :param dictionary (dict): the dictionary to follow the path on
    :param path ([Any]): list of whatever datatypes apply to given dictionary

    :return (Any): the result of following the path
    '''
    if len(path) == 0:
        return dictionary
    
    return follow_path(dictionary[path[0]], path[1:])

def get_path(word, lexemes):
    '''
    Given a dictionary of lexemes for a word's lemma, find the path to the given word's form

    :param word (str): the inflected word to find the path for
    :param lexemes (dict): the dict of lexemes for the given word's lemma

    :return path ([str]): the path to the given word in the lexemes dict
    '''
    for number in NUMBERS:
        for gender in GENDERS:
            if lexemes[number][gender] == word:
                return [number, gender]
        for mood in MOODS:
            for time in TIMES:
                for person in PERSONS:
                    if lexemes[number][mood][time][person] == word:
                        return [number, mood, time, person]
    raise NotInDictionaryException()


def mutate(word, label_param, lemma_to_lex, lemmatizer):
    '''
    Applies mutate to the given word

    :param word (str|[str]): the word for mutate to applied to, could be multiple words (if
        prog or perf previously applied)
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

    :return mutated_word (str): the mutated word
    '''
    if label_param.startswith("CAPITALIZE"):
            if label_param.split("-")[1] == "TRUE":
                return word[0].upper() + word[1:]
            else:
                return word[0].lower() + word[1:]
    
    if isinstance(list, word):
        rest = word[1:]
        word = word[0]
    else:
        rest = None

    if label_param.split('-')[1] == "PROG":
            cur_path = get_path(word, lemma_to_lex[lemmatizer(word)]) # Find current grammatical categories to apply
            estar = follow_path(lemma_to_lex[lemmatizer("estar")], cur_path) # Apply to estar
            if rest:
                return [estar] + [mutate(word, "MOOD-GER", lemma_to_lex, lemmatizer)] + rest
            else:
                return [estar] + [mutate(word, "MOOD-GER", lemma_to_lex, lemmatizer)]
    elif label_param.split('-')[1] == "PERF":
        cur_path = get_path(word, lemma_to_lex[lemmatizer(word)]) # Find current grammatical categories to apply
        haber = follow_path(lemma_to_lex[lemmatizer("haber")], cur_path) # Apply to haber

        if len(label_param.split('-')) > 2 and label_param.split('-')[2] == "SUBJ": # Apply subjunctive if PERF-SUBJ
            haber = mutate(word, "MOOD-SUBJ", lemma_to_lex, lemmatizer)
        
        if rest:
            return [haber] + [mutate(word, "MOOD-PAST-PART", lemma_to_lex, lemmatizer)] + rest
        else:
            return [haber] + [mutate(word, "MOOD-PAST-PART", lemma_to_lex, lemmatizer)]

    lemma = lemmatizer(word)
    lexemes = lemma_to_lex[lemma]
    cur_path = get_path(word, lexemes)
    
    new_path = cur_path.copy()
    new_path[HIERARCHY_DEF[label_param.split('-')[0]]] = '-'.join(label_param.split('-')[1:])

    mutated_word = follow_path(lexemes, new_path)

    if rest:
        return [mutated_word] + rest
    else:
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


def apply_labels(sentence, labels, lemma_to_lex, lemmatizer, vocab, move_type="absolute"):
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
    :param move_type (str): the type of move indexing, either absolute or relative

    :return corrected sentence (str): the corrected sentence
    '''
    sentence = sentence.copy()
    move_labels = []
    for i, label in enumerate(labels):
        sub_labels_queue = deque(BeautifulSoup(label).find_all(True))
        for i in range(len(sub_labels_queue)):
            sub_label = sub_labels_queue.popleft()
            name = sub_label.name
            type = sub_label.get("type", "")
            if name == "MOVE":
                sub_labels_queue.append(sub_label)
            else:
                sentence[i] = apply_label(sentence[i], name, type, lemma_to_lex, lemmatizer, vocab)
        move_labels.append(list(sub_labels_queue))
    movements = [0] * len(sentence)
    original_pos = list(range(len(sentence)))
    for i, moves in move_labels:
        for move in moves:
            word = sentence[i + movements[i]]
            cur_pos = i + movements[i]
            if move_type == "absolute":
                new_pos = int(move.get("type", ""))
            else:
                new_pos = cur_pos + int(move.get("type", ""))
            word_original_pos = original_pos[i]
            for j in range(cur_pos+1, new_pos+1):
                movements[original_pos[j]] -= 1
                sentence[j-1] = sentence[j]
                original_pos[j-1] = original_pos[j]
            sentence[new_pos] = word
            original_pos[new_pos] = word_original_pos
    return sentence


        

