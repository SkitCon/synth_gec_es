# synth_gec_es
[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-yellow.svg)](README-es.md)

SYNthetic Grammatical Error Correction for Spanish (ES) is a system for generating synthetic GEC data for common Spanish grammatical errors to train a GEC model.

Generally to be used to augment smaller high-quality training sets as in *GECToR – Grammatical Error Correction: Tag, Not Rewrite* (2020)

**WORK IN PROGRESS**

## Table of Contents
* [Running the script](#running-the-script)
* [Definitions](#definitions)
* [Mutation Types](#mutation-types)
* [Token-Level Stacking](#token-level-stacking)

## Running the script

The main script is generate.py. This script can be ran as:

```
python generate.py [input file] [output file] [-n/--num-sentences] [number to generate for each origin] [-s/--seq2seq]  [-t/--token] [move-absolute, move-relative, replace]
```

* input file is a path to a file with sentences in Spanish on each line. Note that this script assumes unlabeled data and that each line contains only one sentence.
* output file is optional, defines the output path to place the generated data in. If no output path is supplied, it is placed in the same directory as the input file w/ the name [input file name]_synth.txt.
* --num-sentences is the number of errorful sentences that will be generated from each correct sentence in the supplied corpus. By default 1, but you can set it higher to generate more errorful sentences from one sentence with difference errors.
* --seq2seq means the output synthetic data will include the raw errorful sentence unlabeled for use in a traditional NMT-based seq2seq GEC system (as with BART or T5)
* --token means the output synthetic data will include token-level labels for the errorful sentence (discussed [below](#definitions)) for use in a token-level GEC system (as with GECToR)
  * --token requires at least one associated argument, either move-absolute, move-relative, or replace (if multiple, multiple token-level labels will be supplied using each system
    * move-absolute means the *MOVE* label will be included and it will use *absolute* indices to determine the final position of the token. This final position is calculated **final**, after all other labels are applied.
    * move-relative means the *MOVE* label will be included and it will use *relative* indices to determine the final position of the token. The final is position is calculated **final** and **left-to-right**.
    * replace means the *REPLACE* label will be included instead of *move*

## Definitions

The valid main token-level labels are:
* `<KEEP/>`
  * No change to the token
* `<PRE-ADD token=i/>`
  * Add the given token (based on token index in vocab.txt) immediately before this token
* `<POST-ADD token=i/>`
  * Add the given token (based on token index in vocab.txt) immediately after this token
* `<MUTATE type=i>`
  * Mutate the morphology of this token based on a given type. Major types include CAPITALIZE, GENDER, NUMBER, PERSON, MOOD, and TIME. Further discussion [here](#mutation-types)
* `<REPLACE token=i/>`
  * Replace this token with the given token (based on token index in vocab.txt)
* `<MOVE pos=i/>`
  * Move this token to the postion i (in case of absolute indexing) or move this token i positions (in case of relative indexing).
    * In both cases, indexing is based on the position of tokens after all additions. In the case of relative indexing, moves take place left-to-right

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
<MUTATE type=CAPITALIZE-TRUE/>  <KEEP/>  <MOVE i=1/><MUTATE type=PERSON-2/><MUTATE type=MOOD-SUBJ/>  <KEEP/>  <KEEP/>  <KEEP/>
```

changes the sentence to:

`Espero que tú corras bien.`

This means that this defines a *multi-label classification task* for each token. The expected output for each token is an integer vector with the length of the number of possible labels (30). Each label dimension is binary **except** for the MOVE (value = index), PRE-ADD/POST-ADD (value = token index), and REPLACE (value = token index).

Note this is still a WIP, so I'm happy to take feedback for adjustments of this formatting. This is my first iteration and I do not yet have performance evaluations with this task design.

  
