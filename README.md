# synth_gec_es
[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-yellow.svg)](README-es.md)

SYNthetic Grammatical Error Correction for Spanish (ES) is a system for generating synthetic GEC data for common Spanish grammatical errors to train a GEC model.

Generally to be used to augment smaller high-quality training sets as in *GECToR – Grammatical Error Correction: Tag, Not Rewrite* (2020)

**WORK IN PROGRESS**

## Table of Contents
* [Scripts](#scripts)
  * [generate.py](#generate.py)
  * [parse.py](#parse.py)
  * [label.py](#label.py)
* [Definitions](#definitions)
* [Mutation Types](#mutation-types)
* [Token-Level Stacking](#token-level-stacking)

## Scripts

### generate.py

The main script is generate.py. This script can be ran as:

```
python generate.py INPUT_FILE [OUTPUT_FILE] [-n/--num-sentences] [number to generate for each origin] [-s/--seq2seq]  [-t/--token]
```

This script generates synthetic errorful sentences from well-formed Spanish sentences in a corpus.

* input file is a path to a file with sentences in Spanish on each line. Note that this script assumes unlabeled data and that each line contains only one sentence.
* output file is optional, defines the output path to place the generated data in. If no output path is supplied, it is placed in the same directory as the input file w/ the name [input file name]_synth.txt.
* --num-sentences is the number of errorful sentences that will be generated from each correct sentence in the supplied corpus. By default 1, but you can set it higher to generate more errorful sentences from one sentence with difference errors.
* --seq2seq means the output synthetic data will include the raw errorful sentence unlabeled for use in a traditional NMT-based seq2seq GEC system (as with BART or T5)
* --token means the output synthetic data will include token-level labels for the errorful sentence (discussed [below](#definitions)) for use in a token-level GEC system (as with GECToR)

### parse.py

Ran as:

```
python3 parse.py INPUT_FILE [OUTPUT_FILE] [-d/--dictionary-file] [dictionary file] [-v/--vocab-file] [vocab file]
```

This script takes errorful sentences + token-level labels and parses them into a corrected sentence.

* input file is a path to a file with a sentence on one line, the respective token-level labels on the next line, and a blank line before the next sentence
* output file is optional, defines the output path to place the parsed sentences in. If no output path is supplied, it is placed in the same directory as the input file w/ the name [input file name]_parsed.txt.
* --dictionary-file is the path to the dictionary file which supplies different morphological forms for a word
* --vocab-file is the path to the vocab file containing all words in your model's vocabulary

### label.py

Ran as:

```
python3 label.py INPUT_FILE [OUTPUT_FILE] [-d/--dictionary-file] [dictionary file] [-v/--vocab-file] [vocab file]
```

This script takes errorful sentences + target sentences and translates them into token-level edits using shortest edit distance.

* input file is a path to a file with an errorful sentence on one line, the target sentence on the next, and a blank line before the next sentence
* output file is optional, defines the output path to place the labeled sentences in. If no output path is supplied, it is placed in the same directory as the input file w/ the name [input file name]_labeled.txt.
* --dictionary-file is the path to the dictionary file which supplies different morphological forms for a word
* --vocab-file is the path to the vocab file containing all words in your model's vocabulary

## Definitions

The valid main token-level labels are:
* `<KEEP/>`
  * No change to the token
* `<DELETE/>`
  * Delete the token
* `<PRE-ADD param=i/>`
  * Add the given token (based on token index in vocab.txt) immediately before this token
* `<POST-ADD param=i/>`
  * Add the given token (based on token index in vocab.txt) immediately after this token
* `<MUTATE param=i>`
  * Mutate the morphology of this token based on a given type. Major types include CAPITALIZE, GENDER, NUMBER, PERSON, MOOD, and TIME. Further discussion [here](#mutation-types)
* `<REPLACE param=i/>`
  * Replace this token with the given token (based on token index in vocab.txt)

### Why Both PRE- and POST-ADD?
PRE-ADD and POST-ADD are redundant. In reality, all sentence transformations are obviously possible with only one of these. However, I believe having both possible transformations may allow for more flexibility in error correction. My hypothesis is that allowing a token to be tagged w/ either ADD type will train a more bidirectional understanding of how to build phrases.

For example, lets say we have the incorrect sentence:

  (1) *Él es trabajador muy.

The most natural way to correct this is:

```
<KEEP/> <KEEP/> <PRE-ADD token=muy/> <DELETE/> <KEEP/>
```

which results in:

  (2) Él es muy trabajador.

This is because *muy* is in the same hierarchical grouping as *trabajador*. *muy* modifies *trabajador*.

If we only had POST-ADD (i.e. APPEND), the correction would be:

```
<KEEP/> <POST-ADD token=muy/> <KEEP/> <DELETE/> <KEEP/>
```

While this works fine, there is not a semantic connection between *es* and *muy*.

In a simplified syntax tree we can see this easily:

```
     S
   /   \
  N     VP
  |    /  \
  Él  V    NP
      |   /  \
     es ADV  ADJ
         |    |
        muy  trabajador
```

*muy* and *trabajador* are sisters; whereas, *es* and *muy* are cousins. These dependencies apply in both directions as human language is hierarchical, not sequential.

## Mutation Types
* CAPITALIZE
  * TRUE - capitalizes the word
  * FALSE - uncapitzalizes the word
* GENDER (GÉNERO)
  * MASC - makes the word masculine (if possible)
    * e.g. apuesta + `<MUTATE type=GENDER-MASC/>` = apuesto
  * FEM - makes the word feminine (if possible)
    * e.g. harto + `<MUTATE type=GENDER-MASC/>` = harta
* NUMBER (NÚMERO)
  * SING - makes the word singular
  * PLU - makes the word plural
* PERSON (PERSONA)
  * 1 - makes the word first-person (must be verb)
    * e.g. son + `<MUTATE type=PERSON-1` = somos
  * 2 - makes the word second-person (must be verb)
  * 3 - makes the word third-person (must be verb)
* MOOD (MODO)
  * IND (INDICATIVO) - makes the word indicative in mood (must be verb)
  * POS-IMP (IMPERATIVO AFIRMATIVO) - makes the word positive imperative in mood (must be verb)
  * NEG-IMP (IMPERATIVO NEGATIVO) - makes the word positive imperative in mood (must be verb) (adds *no* before the verb)
  * SUBJ (SUBJUNTIVO) - makes the word subjunctive in mood (must be verb)
  * PROG (PROGRESIVO) - makes the word progressive in mood (must be verb) (adds the correct conjugation of *estar* before the verb)
  * PERF (PERFECTO) - makes the word perfect in mood (must be verb) (adds the correct conjugation of *haber* before the verb)
  * PERF-SUBJ (PERFECTO SUBJUNCTIVO) - makes the word perfect subjunctive in mood (must be verb) (adds the correct conjugation of *haber* before the verb)
  * GER (PARTICIPIO PRESENTE) - makes the word in the gerund form, distinct from progressive which also adds the correct conjugation of *estar* (must be verb)
  * PAST-PART (PARTICIPIO PASADO) - makes the word in the past participle form (must be verb)
  * INF (INFINITIVO) - makes word in the infinitive form (i.e. no mood) (must be verb)
* TIME (TIEMPO)
  * PRES (PRESENTE)
  * PRET (PRETÉTERITO)
  * IMPERF (IMPERFECTO)
  * COND (CONDICIONAL)
  * FUT (FUTURO)

## Token-Label Stacking
You may have noticed that these labels are not mutually exclusive. Some labels are incompatible or redundant together, but many would be expected to operate in conjunction. Therefore, the textual label is formatted with tabs in between each token's label. If there is no tab between two labels, then all of those labels apply to the same token. For example:

```
espero que corre tú bien.
<MUTATE type=CAPITALIZE-TRUE/>  <KEEP/>  <MUTATE type=PERSON-2/><MUTATE type=MOOD-SUBJ/><PRE-ADD token=tú/> <DELETE/>  <KEEP/>  <KEEP/>
```

changes the sentence to:

`Espero que tú corras bien.`

This means that this defines a *multi-label classification task* for each token. The expected output for each token is an integer vector with the length of the number of possible labels (30). Each label dimension is binary **except** PRE-ADD/POST-ADD (value = token index) and REPLACE (value = token index).

Note this is still a WIP, so I'm happy to take feedback for adjustments of this formatting. This is my first iteration and I do not yet have performance evaluations with this task design.

  
