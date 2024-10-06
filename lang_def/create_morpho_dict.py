'''
File: create_morpho_dict.py
Author: Amber Converse
Purpose: This file creates the morphological dict for surface realization from 2 files,
    the first file is a JSONL file parsed from ES Wikitionary and the second is a CSV containing
    the verb forms for Spanish verbs.
'''

import json
import argparse
import pandas as pd
from collections import defaultdict

def simplify_pos(pos):
    if pos in ["unknown", "abbrev", "num", "suffix", "phrase", "participle", "intj", "name"]:
        return None
    
    if pos == "noun":
        return "NOUN"
    elif pos == "pron":
        return "PRONOUN"
    elif pos in ["proverb", "verb"]:
        return "VERB"
    elif pos == "article":
        return "ARTICLE"
    elif pos == "adj":
        return "ADJ"
    elif pos == "adv":
        return "adv"
    
def get_category(tags, raw_tags, category=[]):
    
    if "number" in category:
        if "singular" in tags:
            yield "SING"
        elif "plural" in tags:
            yield "PLUR"
        elif "Singular" in raw_tags or "Singularia" in raw_tags:
            yield "SING"
        elif "plural" in raw_tags or "Pluralia" in raw_tags:
            yield "PLUR"
        else:
            yield "SING"
    if "gender" in category:
        if "masculine" in tags:
            yield "MASC"
        elif "feminine" in tags:
            yield "FEM"
        elif "masculino" in raw_tags:
            yield "MASC"
        elif "Femenino" in raw_tags:
            yield "FEM"
        else:
            yield "MASC"
    if "case" in category:
        if "Nominativo" in raw_tags:
            yield "NOM"
        elif "Acusativo" in raw_tags:
            yield "ACC"
        elif "Dativo" in raw_tags:
            yield "DAT"
        else:
            yield "NOM"
    if "definite" in category:
        if "indefinite" in tags:
            yield "IND"
        else:
            yield "DEF"

def translate_mood(mood):
    if mood == "Indicativo":
        return "IND"
    elif mood == "Subjuntivo":
        return "SUB"
    elif mood == "Imperativo Negativo":
        return "NEG-IMP"
    elif mood == "Imperativo Afirmativo":
        return "POS-IMP"
    
def translate_time(time):
    if time in ["Pluscuamperfecto", "Presente perfecto", "Futuro perfecto", "Pretérito anterior"]:
        return None
    
    if time == "Presente":
        return "PRES"
    elif time == "Pretérito":
        return "PRET"
    elif time == "Imperfecto":
        return "IMP"
    elif time == "Condicional":
        return "CND"
    elif time == "Futuro":
        return "FUT"
    
def get_col_name(number, person, formality=None):
    if formality:
        if formality == "FORM":
            person = "3"
    
    return f"form_{person}{number[0].lower()}"
        
def fill_dict(row):
    lemma = row["infinitive"]
    mood = translate_mood(row["mood"])
    time = translate_time(row["tense"])
    if not time: # if invalid time, skip
        return
    for number in ["SING", "PLUR"]:
        for person in ["1", "2", "3"]:
            if not person == "2":
                col_name = get_col_name(number, person)
                lemma_to_morph[lemma][number][mood][time][person]["SURFACE_FORM"] = row[col_name]
    
    # For special versions
    for mood in ["GER", "PAST-PART", "INF"]:
        if mood == "GER":
            lemma_to_morph[lemma][mood]["SURFACE_FORM"] = row["gerund"]
        elif mood == "PART-PART":
            lemma_to_morph[lemma][mood]["SURFACE_FORM"] = row["pastparticiple"]
        elif mood == "INF":
            lemma_to_morph[lemma][mood]["SURFACE_FORM"] = lemma

def create_default_dict(wiki_file, verb_forms_file, output_file):
    global lemma_to_morph

    words = []
    with open(wiki_file, 'r') as f:
        for line in f.readlines():
            words.append(json.loads(line))

    words = [word for word in words if "lang_code" in word and word["lang_code"] == "es"]
    words = [word for word in words if "forms" in word]

    nested_dict = lambda: defaultdict(nested_dict)
    lemma_to_morph = nested_dict()

    for word in words:
        lemma = word["word"]
        pos = simplify_pos(word["pos"])
        if not pos or pos == "VERB":
            continue
        
        for form in word["forms"]:
            
            tags = form["tags"] if "tags" in form else []
            raw_tags = []
            if "raw_tags" in form:
                for raw_tag in form["raw_tags"]:
                    raw_tags += raw_tag.split()
            
            number, gender = get_category(tags, raw_tags, category=["number", "gender"])
            
            if pos in ["NOUN", "ADJ", "ADV"]:
                lemma_to_morph[lemma][pos][number][gender]["SURFACE_FORM"] = form["form"]
                continue
            elif pos == "PRONOUN":
                case = next(get_category(tags, raw_tags, category=["case"]))
            else:
                definite = next(get_category(tags, raw_tags, category=["definite"]))
            
            if pos == "PRONOUN":
                lemma_to_morph[lemma][pos][number][gender][case]["SURFACE_FORM"] = form["form"]
                continue
            elif pos == "ARTICLE":
                lemma_to_morph[lemma][pos][number][gender][definite]["SURFACE_FORM"] = form["form"]
                continue

    df = pd.read_csv(verb_forms_file)

    df.apply(fill_dict, axis=1)

    with open(output_file, 'w') as f:
        json.dump(lemma_to_morph, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input1", help="path to Wikitionary JSONL parse")
    parser.add_argument("input2", help="path to verb forms CSV")
    parser.add_argument("output", help="output_path")

    args = parser.parse_args()

    if not args.output:
        output_file = "morpho_dict.csv"
    else:
         output_file = args.output

    create_default_dict(args.input1, args.input2, output_file)