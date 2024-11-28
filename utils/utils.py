'''
File: utils.py
Author: Amber Converse
Purpose: Contains a variety of functions for use in generating synthetic errorful sentences in
    Spanish
'''
import json
import re
from bs4 import BeautifulSoup
import spacy
from spacy.tokens import Doc
from transformers import AutoTokenizer
from unidecode import unidecode

from utils.custom_errors import InvalidLabelException
from utils.custom_errors import NotInDictionaryException

CONTEXT_WINDOW = 1 # Context window for spacy analysis of edited tokens

DEFAULT_PATH_FOR_POS = {"NOUN": ["NOUN", "SING", "MASC", "SURFACE_FORM"],
                         "ADJ": ["ADJ", "SING", "MASC", "SURFACE_FORM"],
                         "ADV": ["ADV", "SING", "MASC", "SURFACE_FORM"],
                         "PRONOUN": ["PRONOUN", "SING", "MASC", "NOM", "SURFACE_FORM"],
                         "PERSONAL_PRONOUN": ["PERSONAL_PRONOUN", "SING", "MASC", "NOM", "BASE", "NO", "SURFACE_FORM"],
                         "ARTICLE": ["ARTICLE", "SING", "MASC", "DEF", "SURFACE_FORM"],
                         "VERB": ["VERB", "SING", "IND", "PRES", "3", "SURFACE_FORM"],
                         "X": ["X", "SURFACE_FORM"]}

POS = ["NOUN", "PRONOUN", "PERSONAL_PRONOUN", "VERB", "ARTICLE", "ADJ", "ADV"]
GENDERS = ["MASC", "FEM"] # No non-binary :( RAE dice que no
NUMBERS = ["SING", "PLUR"]
DEFINITE = ["DEF", "IND"]
CASE = ["NOM", "ACC", "DAT"]
MOODS = ["IND", "SUB", "GER", "PAST-PART", "INF"]
TIMES = ["PRES", "PRET", "IMP", "CND", "FUT"]
PERSONS = ["1", "2", "3"]
PRONOUN_TYPES = ["BASE", "CLITIC"]
REFLEXIVE = ["YES", "NO"]

CATEGORIES_FOR_POS = {"NOUN": ["POS", "NUMBER", "GENDER"],
                      "ADJ": ["POS", "NUMBER", "GENDER"],
                      "ADV": ["POS", "NUMBER", "GENDER"],
                      "PRONOUN": ["POS", "NUMBER", "GENDER", "CASE"],
                      "PERSONAL_PRONOUN": ["POS", "NUMBER", "GENDER", "CASE", "PRONOUN_TYPE", "REFLEXIVE"],
                      "ARTICLE": ["POS", "NUMBER", "GENDER", "DEFINITE"],
                      "VERB": ["POS", "NUMBER", "MOOD", "TIME", "PERSON"],
                      "X": ["POS"]}

AVAILABLE_TYPES = {"POS": POS,
                   "GENDER": GENDERS,
                   "NUMBER": NUMBERS,
                   "DEFINITE": DEFINITE,
                   "CASE": CASE,
                   "MOOD": MOODS,
                   "TIME": TIMES,
                   "PERSON": PERSONS,
                   "PRONOUN_TYPE": PRONOUN_TYPES,
                   "REFLEXIVE": REFLEXIVE}


HIERARCHY_DEF = {"POS": 0,
                    "NUMBER": 1,
                    "MOOD": [1,2],
                        "GENDER": 2,
                            "CASE": 3,
                                "PRONOUN_TYPE": 4,
                                    "REFLEXIVE": 5,
                            "DEFINITE": 3,
                            "TIME": 3,
                                "PERSON": 4,
                                    "FORMALITY": 5}
UPOS_TO_SIMPLE = {"ADJ": "ADJ",
                  "ADP": "X",
                  "ADV": "ADV",
                  "AUX": "VERB",
                  "CCONJ": "X",
                  "DET": "ARTICLE",
                  "INTJ": "X",
                  "NOUN": "NOUN",
                  "NUM": "X",
                  "PART": "X",
                  "PRON": "PRONOUN",
                  "PROPN": "NOUN",
                  "PUNCT": "X",
                  "SCONJ": "X",
                  "SYM": "X",
                  "VERB": "VERB",
                  "SPACE": "X",
                  "X": "X"}

def load_vocab(filename):
    with open(filename, 'r') as f:
        return [token.strip() for token in f]
    
def load_morpho_dict(filename):
    '''
    Loads the lexical dictionary in filename
    
    :param filename (str): the path to the dictionary file

    :return lemma_to_morph (dict): a dictionary that maps the base form of a word to all of its
        morphological forms.
    '''
    with open(filename, 'r') as f:
        lemma_to_morph = json.load(f)
    return lemma_to_morph

def create_vocab_index(vocab):
    return {word: i for i, word in enumerate(vocab)}

def load_modified_nlp(model_path="es_dep_news_trf", tokenizer_path="dccuchile/bert-base-spanish-wwm-cased"):
    nlp = spacy.load(model_path)
    custom_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    nlp.tokenizer = lambda text: Doc(nlp.vocab, words=custom_tokenizer.tokenize(text))
    return nlp

def clean_text(text):
    text = re.sub("\s+", ' ', text).strip()
    replacements = [(r"[“”]", "\""), \
                    (r"[‘’]", "'"), \
                    (r"…", "..."), \
                    (r"[\"\']", '')] # This one is for my sanity. Let's pretend quotes don't exist :)
    for replacement in replacements:
        text = re.sub(replacement[0], replacement[1], text)
    return text.strip()

def process_lemma(raw_lemma):
    '''
    To solve weird lemmatization problem where something like "empezábamos" has the lemma: "empezár"
    '''
    if raw_lemma.endswith('r'):
        return raw_lemma[:-2] + unidecode(raw_lemma[-2]) + raw_lemma[-1]
    else:
        return raw_lemma

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
    
    try:
        return follow_path(dictionary[path[0]], path[1:])
    except KeyError: # Failure to resolve, most likely due to change in POS or missing entry
        if path[0] in POS: # If we are at the beginning of the path, we have some options
            cur_pos = path[0]
            if cur_pos != "VERB": # if it not a verb, we can try looking at other POS
                tries = ["NOUN", "ADJ", "ADV"]
                if cur_pos in tries:
                    tries.remove(cur_pos)
                for pos_try in tries:
                    try:
                        return follow_path(dictionary[pos_try], path[1:3] + ["SURFACE_FORM"])
                    except KeyError: # If there is a KeyError just keep going
                        pass
                # Try other POS
                if cur_pos != "PRONOUN":
                    try:
                        if cur_pos == "PERSONAL_PRONOUN":
                            return follow_path(dictionary["PRONOUN"], path[1:4] + ["SURFACE_FORM"])
                        else:
                            return follow_path(dictionary["PRONOUN"], path[1:3] + ["NOM", "SURFACE_FORM"])
                    except KeyError:
                        pass
                if cur_pos != "PERSONAL_PRONOUN":
                    try:
                        if cur_pos == "PRONOUN":
                            return follow_path(dictionary["PERSONAL_PRONOUN"], path[1:4] + ["CLITIC", "NO", "SURFACE_FORM"])
                        else:
                            return follow_path(dictionary["PERSONAL_PRONOUN"], path[1:3] + ["NOM", "CLITIC", "NO", "SURFACE_FORM"])
                    except KeyError:
                        pass
                    try:
                        if cur_pos == "PRONOUN":
                            return follow_path(dictionary["PERSONAL_PRONOUN"], path[1:4] + ["BASE", "NO", "SURFACE_FORM"])
                        else:
                            return follow_path(dictionary["PERSONAL_PRONOUN"], path[1:3] + ["NOM", "BASE", "NO", "SURFACE_FORM"])
                    except KeyError:
                        pass
                if cur_pos != "ARTICLE":
                    try:
                        return follow_path(dictionary["ARTICLE"], path[1:3] + ["DEF", "SURFACE_FORM"])
                    except KeyError:
                        pass
                    try:
                        return follow_path(dictionary["ARTICLE"], path[1:3] + ["IND", "SURFACE_FORM"])
                    except KeyError:
                        pass
            # As a last resort, try the default path for this POS
            new_dict = dictionary[path[0]]
            path = DEFAULT_PATH_FOR_POS[path[0]][1:]
            return follow_path(new_dict, path)
        else: # Too deep to resolve, go back up until back at the beginning of the path
            raise KeyError

{"NOUN": ["NOUN", "SING", "MASC", "SURFACE_FORM"],
                         "ADJ": ["ADJ", "SING", "MASC", "SURFACE_FORM"],
                         "ADV": ["ADV", "SING", "MASC", "SURFACE_FORM"],
                         "PRONOUN": ["PRONOUN", "SING", "MASC", "NOM", "SURFACE_FORM"],
                         "PERSONAL_PRONOUN": ["PERSONAL_PRONOUN", "SING", "MASC", "NOM", "BASE", "NO", "SURFACE_FORM"],
                         "ARTICLE": ["ARTICLE", "SING", "MASC", "DEF", "SURFACE_FORM"],
                         "VERB": ["VERB", "SING", "IND", "PRES", "3", "SURFACE_FORM"],
                         "X": ["X", "SURFACE_FORM"]}

def get_path(token):
    '''
    Given a dictionary of morphology for a word's lemma, find the path to the given word's form

    :param word (Doc): the inflected word to find the path for

    :return path ([str]): the path to the given word in the morphology dict
    '''
    pos = UPOS_TO_SIMPLE[token.pos_]
    if pos == "PRONOUN": # Determine if needs extra categories
        if token.text.lower() in {"yo", "me", "mí", "tú", "te", "ti", "usted", "le", "lo", "la", "se", \
                                  "él", "ella", "nosotros", "nosotras", "nos", "ustedes", "les", "los", \
                                    "las", "vosotros", "vosotras", "os"}:
            pos = "PERSONAL_PRONOUN"
        else:
            pos = "PRONOUN"

    number = token.morph.get("Number")[0].upper() if token.morph.get("Number") else "SING"
    
    gender = token.morph.get("Gender")[0].upper() if token.morph.get("Gender") else "MASC"

    case = token.morph.get("Case")
    if not case or "Nom" in case: # Prioritize NOM over ACC and DAT
        case = "NOM"
    elif "Acc" in case: # Prioritize ACC over DAT
        case = "ACC"
    else:
        case = "DAT"

    # Clitic pronouns are designated as "Npr" or not after a preposition, we use this to distinguish them
    pronoun_type = token.morph.get("PrepCase")[0] if token.morph.get("PrepCase") else "BASE"
    # Dative clitic pronouns are not designated as "Npr", non-clitic pronouns do not have dative morphology in Spanish,
    # so we can assume a dative personal pronoun is clitic
    if pronoun_type == "Npr" or case == "DAT":
        pronoun_type = "CLITIC"
    else:
        pronoun_type = "BASE"
    
    reflexive = token.morph.get("Reflex")[0] if token.morph.get("Reflex") else "NO"
    # Once again, weird case with dative clitic pronouns, something like "se" in "se lo doy" has reflexive morphology, but is not tagged as reflexive
    # by spaCy. Instead it is given the preposition case "Npr", which dative clitic pronouns do not normally have, except when they have reflexive morphology.
    # I can see the logic in this decision (dative = implicit preposition unless it is in the special reflexive form), but it was difficult to debug.
    if pronoun_type == "CLITIC" and case == "DAT" and (token.morph.get("PrepCase")[0] if token.morph.get("PrepCase") else "BASE") == "Npr":
        reflexive = "YES"

    definite = token.morph.get("Definite")[0].upper() if token.morph.get("Definite") else "DEF"
   
    mood = token.morph.get("Mood")[0] if token.morph.get("Mood") else "Ind"
    verb_form = token.morph.get("VerbForm")[0] if token.morph.get("VerbForm") else "Fin"
    time = token.morph.get("Tense")[0] if token.morph.get("Tense") else "Pres"

    # Handle disagreement over whether conditional is a mood or a tense, I define it as a tense
    if mood == "Cnd":
        mood = "Ind"
        time = "Cnd"
    
    if verb_form == "Inf":
        mood = "INF"
    elif verb_form == "Part":
        mood = "PAST-PART"
    elif verb_form == "Ger":
        mood = "GER"
    else:
        mood = mood.upper()

    # "PRES", "PRET", "IMPERF", "CND", "FUT"
    if time == "Past":
        time = "PRET"
    else:
        time = time.upper()

    person = str(token.morph.get("Person")[0]) if token.morph.get("Person") else "3"
    
    if pos in ["NOUN", "ADJ", "ADV"]:
        return [pos, number, gender, "SURFACE_FORM"]
    elif pos == "PRONOUN":
        return [pos, number, gender, case, "SURFACE_FORM"]
    elif pos == "PERSONAL_PRONOUN":
        return [pos, number, gender, case, pronoun_type, reflexive, "SURFACE_FORM"]
    elif pos == "ARTICLE":
        return [pos, number, gender, definite, "SURFACE_FORM"]
    elif pos == "VERB":
        if mood in ["GER", "PAST-PART", "INF"]:
            return [pos, mood, "SURFACE_FORM"]
        else:
            return [pos, number, mood, time, person, "SURFACE_FORM"]
    else:
        return [pos, "SURFACE_FORM"]

def get_new_path(token, label_param):
    cur_path = get_path(token)

    # Ensure param is compatible with POS
    category_to_change = label_param.split('-')[0]
    if not category_to_change in CATEGORIES_FOR_POS[cur_path[0]]: # Mismatch, need to repair path
        new_pos = None
        new_cur_path = None
        for pos, categories in CATEGORIES_FOR_POS.items():
            if category_to_change in categories:
                new_pos = pos
                new_cur_path = DEFAULT_PATH_FOR_POS[pos]
                break
        if not new_cur_path:
            raise InvalidLabelException(f"Mutation \"{label_param}\" does not exist.")
        # Transfer what is possible
        overlap = [i for i in range(1, min(len(CATEGORIES_FOR_POS[cur_path[0]]), len(CATEGORIES_FOR_POS[new_pos])))
                   if CATEGORIES_FOR_POS[cur_path[0]][i] == CATEGORIES_FOR_POS[new_pos][i]]
        for i in overlap:
            new_cur_path[i] = cur_path[i]
        cur_path = new_cur_path

    # Going from special verb to normal verb, apply default path
    if cur_path[0] == "VERB" and len(cur_path) == 3 and not label_param in ["MOOD-INF", "MOOD-GER", "MOOD-PAST-PART"]:
        cur_path = DEFAULT_PATH_FOR_POS["VERB"]
    
    new_path = cur_path.copy()
    index_of_change = HIERARCHY_DEF[label_param.split('-')[0]]
    if isinstance(index_of_change, list): # Special case for verbs when changing mood
        if new_path[2] == "SURFACE_FORM": # Special verb case
            index_of_change = 1
        else: # Normal verb case
            index_of_change = 2
    
    # Going from normal verb to special verb, eliminate extra categories
    if label_param in ["MOOD-INF", "MOOD-GER", "MOOD-PAST-PART"]:
        new_path = ["VERB", '-'.join(label_param.split('-')[1:]), "SURFACE_FORM"]
    else:
        new_path[index_of_change] = '-'.join(label_param.split('-')[1:])
    
    return new_path

def restore_nlp(sentence, token, token_idx, nlp):
    left_context = []
    for segment in sentence[max(0, token_idx - CONTEXT_WINDOW):token_idx]:
        if not isinstance(segment, list):
            segment = [segment]
        for cur_token in segment:
            left_context.append(cur_token)
    # Find main word for any sub-words
    cur_left_start = max(0, token_idx - CONTEXT_WINDOW) - 1
    while cur_left_start >= 0 and len(left_context) > 0 and str(left_context[0]).startswith("##"):
        prepend_word = sentence[cur_left_start]
        if not isinstance(prepend_word, list):
            prepend_word = [prepend_word]
        left_context = prepend_word + left_context
        cur_left_start -= 1
    right_context = []
    for segment in sentence[token_idx+1:min(len(sentence), token_idx + CONTEXT_WINDOW + 1)]:
        if not isinstance(segment, list):
            segment = [segment]
        for cur_token in segment:
            right_context.append(cur_token)
    with_context = left_context + token + right_context
    all_strs = [str(cur_token) for cur_token in with_context]
    combined = []
    for cur_token in all_strs:
        if cur_token.startswith("##") and len(combined) > 0: # Tokenized as parts, needs to be combined with previous word
            combined[-1] += cur_token[2:]
        else:
            combined.append(cur_token)
    res = [token for token in nlp(' '.join(combined))][len(left_context):len(token) + len(left_context)]
    # If tokenization changes for sub-words with context, we can't really add context, so we have to just do the word by itself
    # and hope for the best
    if [str(cur_token) for cur_token in res] != [str(cur_token for cur_token in token)]:
        all_strs = [str(cur_token) for cur_token in token]
        combined = []
        extra_token = False
        for cur_token in all_strs:
            if cur_token.startswith("##") and len(combined) > 0: # Tokenized as parts, needs to be combined with previous word
                combined[-1] += cur_token[2:]
            # If the below statement triggers, there is a sub-word is at the beginning of the sequence, but we can't grab the
            # other main word before it because we already know it changes tokenization!! To solve this, I just prepend 1 to the
            # sub-word. This is because 1 + sub-word will always tokenize as [1, ##sub-word]. This works because we take advantage
            # of the logic for tokenizing numeric suffixes (in Spanish this is something 1ro/1o for primero, the English equivalent
            # is 1st for first). This is a very weird solution, I know. This solution is only tested for BERT-like tokenizers.
            elif cur_token.startswith("##"):
                combined.append("1" + cur_token[2:])
                extra_token = True
            else:
                combined.append(cur_token)
        if extra_token: # Added 1 to the beginning to force tokenization in parts, need to remove it
            return [token for token in nlp(' '.join(combined))][1:]
        else:
            return [token for token in nlp(' '.join(combined))]
    else:
        return res



def mutate(token, label_param, lemma_to_morph, nlp):
    '''
    Applies mutate to the given word

    :param token (Token): the word for mutate to applied to
    :param label_param (int|str): the parameter for the label (if applicable)
    :param lemma_to_morph (dict): a dictionary that maps the base form of a word to all of its
        lexical forms in hierarhical format of
            LEMMA
            |- NUMBER
                |- GENDER,
                |- MOOD,
                    |- TIME
                        |- PERSON
    :param nlp (spaCy model): the model for tokenization, lemmatization, and morphology of a word/sentence
    

    :return mutated_word (str): the mutated word
    '''

    if type(token) == spacy.tokens.doc.Doc:
        token = token[0]

    if label_param.startswith("CAPITALIZE"):
        if label_param.split("-")[1] == "TRUE":
            return token.text[0].upper() + token.text[1:]
        else:
            return token.text.lower()

    if label_param.split('-')[1] == "PROG":
            cur_path = get_path(token, lemma_to_morph[process_lemma(token.lemma_)]) # Find current grammatical categories to apply
            estar = follow_path(lemma_to_morph["estar"], cur_path) # Apply to estar
            return [nlp(estar)[0]] + [mutate(token, "MOOD-GER", lemma_to_morph, nlp)]
    elif label_param.split('-')[1] == "PERF":
        cur_path = get_path(token, lemma_to_morph[process_lemma(token.lemma_)]) # Find current grammatical categories to apply
        haber = follow_path(lemma_to_morph["haber"], cur_path) # Apply to haber

        if len(label_param.split('-')) > 2 and label_param.split('-')[2] == "SUBJ": # Apply subjunctive if PERF-SUBJ
            haber = mutate(nlp(haber)[0], "MOOD-SUBJ", lemma_to_morph)
        
        return [haber] + [mutate(token, "MOOD-PAST-PART", lemma_to_morph)]

    lemma = process_lemma(token.lemma_)
    morphology = lemma_to_morph[lemma]
    new_path = get_new_path(token, label_param)

    mutated_word = follow_path(morphology, new_path)

    return mutated_word


def apply_label(word, label, label_param, lemma_to_morph, vocab, nlp):
    '''
    Apply label to a word.

    :param word ([Token]): List of words for the label to applied to
    :param label (str): the label to be applied to word
    :param label_param (int|str): the parameter for the label (if applicable)
    :param lemma_to_morph (dict): a dictionary that maps the base form of a word to all of its
        lexical forms in hierarhical format
    :param vocab ([str]): a list of all words in dictionary, aligned with model vocabulary
    :param nlp (spaCy model): the model for tokenization, lemmatization, and morphology of a word/sentence

    :return corrected_word (str): the corrected word based on label
    '''
    if label not in ["KEEP", "DELETE", "ADD", "COPY", "MUTATE", "REPLACE"]:
        raise InvalidLabelException(f"\"{label}\" is not a valid label.")
    
    if label == "KEEP":
        return word
    elif label == "DELETE":
        return []
    elif label == "ADD":
        return [vocab[label_param]] + word
    elif label == "MUTATE":
        res = [mutate(word[0], label_param, lemma_to_morph, nlp)] + word[1:] if len(word) > 1 \
                else [mutate(word[0], label_param, lemma_to_morph, nlp)]
        if not isinstance(res, list):
            res = [res]
        return res
    elif label == "REPLACE":
        return [vocab[label_param]]


def apply_labels(doc, labels, lemma_to_morph, vocab, nlp):
    '''
    Applies the token-level GEC labels to the given sentence.

    sentence and labels must be the same length.

    :param doc (Doc): spaCy doc of the sentence
    :param labels ([str]): list of token-level labels to be applied to errorful sentence for correction
    :param lemma_to_morph (dict): a dictionary that maps the base form of a word to all of its
        lexical forms in hierarhical format of
            LEMMA
            |- NUMBER
                |- GENDER,
                |- MOOD,
                    |- TIME
                        |- PERSON
    :param lemmatizer (func(str)): a function that takes a word as input and returns the lemma
    :param vocab ([str]): a list of all words in dictionary, aligned with model vocabulary
    :param nlp (spaCy model): the model for tokenization, lemmatization, and morphology of a word/sentence

    :return corrected sentence (str): the corrected sentence
    '''
    sentence = [[token] for token in doc]
    sentence = sentence + [[nlp("EOS")[0]]] # Append [EOS]
    for i, label in enumerate(labels):
        for sub_label in BeautifulSoup(label, features="html.parser").find_all(True):
            name = sub_label.name.upper()
            param = sub_label.get("param", "").upper()
            param = int(param) if param.isnumeric() else param
            if len(name.split('-')) < 2 or name.split('-')[0] != "COPY":
                sentence[i] = apply_label(sentence[i], name, param, lemma_to_morph, vocab, nlp)
                
                # Restore spacy for added words, add context because spacy gets confused with singular words
                sentence[i] = restore_nlp(sentence, sentence[i], i, nlp)

            elif name == "COPY-ADD":
                sentence[i] = [doc[param]] + sentence[i]
            elif name == "COPY-REPLACE":
                sentence[i] = [doc[param]]
    
    # Run copies
    # for i, token_copies in enumerate(copies):
    #     for copy in token_copies:
    #         name = copy.name.upper()
    #         param = int(copy.get("param", -1))
    #         cur_token = [sentence[i]] if isinstance(sentence[i], spacy.Token) else sentence[i]
    #         if name.split('-')[0] == "PRE":
    #             sentence[i] = sentence[param] + cur_token
    #         else:
    #             sentence[i] = cur_token + sentence[param]

    # Run deletes
    # for i, delete in enumerate(deletes):
    #     if delete:
    #         sentence[i] = nlp("")

    corrected_sentence = ""
    first_token = True
    no_space = False
    for token in sentence:
        if not isinstance(token, list):
            token = [token]
        for cur_token in token:
            if re.match("^[\.\,\?\!\)\}\]\-\:\;]$", cur_token.text) or first_token or no_space:
                corrected_sentence += cur_token.text
                first_token = False
                no_space = False
            elif re.match("^\#\#.*$", cur_token.text): # Sub-word
                corrected_sentence += cur_token.text[2:]
                first_token = False
                no_space = False
            else:
                corrected_sentence += " " + cur_token.text
            
            if re.match("^[\¿\¡\(\{\[\"]$", cur_token.text): # Handle beginning punctuation
                no_space = True
    
    if corrected_sentence.endswith("E"): # Remove end token
        corrected_sentence = corrected_sentence[:-2]
    
    return corrected_sentence