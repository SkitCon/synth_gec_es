# synth_gec_es
[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-yellow.svg)](README-es.md)

**version 0.5.3**

SYNTHetic Grammatical Error Correction for Spanish (ES) is a system for generating synthetic GEC data for common Spanish grammatical errors to train a GEC model.

To be used to augment smaller high-quality training sets as in *GECToR – Grammatical Error Correction: Tag, Not Rewrite* (2020)

**WORK IN PROGRESS**
All scripts are working, but may still contain minor bugs. In general, if a script fails, an error message will print and it will continue with the rest of the file, ignoring that sentence.

Required libraries for all scripts:
```
bs4 == 0.0.2
spacy == 3.7.6
unidecode == 1.3
transformers == 4.31.0
```

## Table of Contents
* [Scripts](#scripts)
  * [generate.py](#generate.py)
  * [decode.py](#decode.py)
  * [label.py](#label.py)
* [Definitions](#definitions)
* [Mutation Types](#mutation-types)
* [Token-Level Stacking](#token-level-stacking)

## Scripts

Note: If using the same file organization as the repo (i.e. if you cloned this repo), do not worry about the --dict_file or --vocab_file args.

### generate.py

The main script is generate.py. This script can be ran as:

```
python generate.py INPUT_FILE OUTPUT_FILE ERROR_FILE_1 ERROR_FILE_2 ... ERROR_FILE_N [--min/-min_error] [minimum number of errors in a sentence] [-max/--max_error] [maximum number of errors in a sentence] [-d/--dict_file] [dictionary file] [--vocab_file] [vocab file] [--seed] [seed] [-n/--num-sentences] [number of sentences to generate for each original] [-t/--token] [--verify] [-v/--verbose] [-sw/--silence_warnings] [--strict]
```

This script generates synthetic errorful sentences from well-formed Spanish sentences in a corpus.

* input file is a path to a file with one sentence in Spanish on a line with a blank line in between each sentence. Note that this script assumes unlabeled data and that each line contains only one sentence.
* output file is optional, defines the output path to place the generated data in. If no output path is supplied, it is placed in the same directory as the input file w/ the name [input file name]_synth.txt.
* error files are json files with lists of errors to apply
* min_error is the minimum number of errors to be generated for a sentence
* max_error is the maximum number of errors to be generated for a sentence
* --dict_file is the path to the dictionary file which supplies different morphological forms for a word
* --vocab_file is the path to the vocab file containing all words in your model's vocabulary
* --seed is the seed for random generation
* --num-sentences is the number of errorful sentences that will be generated from each correct sentence in the supplied corpus. By default 1, but you can set it higher to generate more errorful sentences from one sentence with difference errors.
* --token means the output synthetic data will include token-level labels for the errorful sentence (discussed [below](#definitions)) for use in a token-level GEC system (as with GECToR)
* --verify means the generated token labels will be verified using the decode algorithm to ensure that the result matches the correct sentence. Note that this will generally double the time to label, but guarantees that the labels are valid
* --verbose means debugging code will print
* --silence_warnings means warnings will not be printed, such as a mutation being replaced by a replace
* --strict means the script will be strict about which sentences are included in the output file. This has the primary effect of excluding sentences where a MUTATE verification failed and a REPLACE had to be used to repair the labels.

### decode.py

Ran as:

```
python3 decode.py INPUT_FILE [OUTPUT_FILE] [-d/--dict_file] [dictionary file] [--vocab_file] [vocab file] [-sw/--silence_warnings]
```

This script takes errorful sentences + token-level labels and decodes them into a corrected sentence.

* input file is a path to a file with a sentence on one line, the respective token-level labels on the next line, and a blank line before the next sentence
* output file is optional, defines the output path to place the parsed sentences in. If no output path is supplied, it is placed in the same directory as the input file w/ the name [input file name]_parsed.txt.
* --dict_file is the path to the dictionary file which supplies different morphological forms for a word
* --vocab_file is the path to the vocab file containing all words in your model's vocabulary
* --silence_warnings means warnings will not be printed, such as a mutation being replaced by a replace

### label.py

Ran as:

```
python3 label.py INPUT_FILE [OUTPUT_FILE] [-d/--dictionary-file] [dictionary file] [--vocab_file] [vocab file] [--verify] [-v/--verbose] [-sw/--silence_warnings] [--strict]
```

This script takes errorful sentences + target sentences and translates them into token-level edits using shortest edit distance.

* input file is a path to a file with an errorful sentence on one line, the target sentence on the next, and a blank line before the next sentence
* output file is optional, defines the output path to place the labeled sentences in. If no output path is supplied, it is placed in the same directory as the input file w/ the name [input file name]_labeled.txt.
* --dictionary-file is the path to the dictionary file which supplies different morphological forms for a word
* --vocab-file is the path to the vocab file containing all words in your model's vocabulary
* --verify means the generated token labels will be verified using the decode algorithm to ensure that the result matches the correct sentence. Note that this will generally double the time to label, but guarantees that the labels are valid
* --verbose means debugging code will print
* --silence_warnings means warnings will not be printed, such as a mutation being replaced by a replace
* --strict means the script will be strict about which sentences are included in the output file. This has the primary effect of excluding sentences where a MUTATE verification failed and a REPLACE had to be used to repair the labels.

## Definitions

In order to accomodate adding words to the end of the sentence (because adds are prepends), sentences are appended with \[EOS\] and these tokens are given token-level labels. This is handled by the generation code, so you do not need to apply this yourself. However, when training a token classification model, an \[EOS\] token must be added to the end if not already.

The valid main token-level labels are:
* `<KEEP/>`
  * No change to the token
* `<DELETE/>`
  * Delete the token
* `<ADD param=i/>`
  * Add the given token (based on token index in vocab.txt) immediately before this token
* `<COPY-REPLACE param=i/>`
  * Replaces this token with the given token (based on token index)
* `<COPY-ADD param=i/>`
  * Add the given token (based on token index) immediately before this token
* `<MUTATE param=i/>`
  * Mutate the morphology of this token based on a given type. Major types include CAPITALIZE, GENDER, NUMBER, PERSON, MOOD, and TIME. Further discussion [here](#mutation-types)
* `<REPLACE param=i/>`
  * Replace this token with the given token (based on token index in vocab.txt)

## Mutation Types

Note: Much more detail needed to be added for personal pronouns and the resolution dict needed to be manually created for the detail required. For example, resolution from *él* to *las* or *les* to *se* requires specification of the morphology of clitic pronouns and reflexivity.
* CAPITALIZE
  * TRUE - capitalizes the word
  * FALSE - uncapitzalizes the word
* POS
  * Note: Because the morphology from different parts-of-speech are not generally compatible, default morphology is defined for each POS
  * NOUN - changes the part-of-speech to noun
    * Default morphology = SING, MASC
  * PRONOUN changes the part-of-speech to pronoun
    * Default morphology = SING, MASC, NOM
  * VERB changes the part-of-speech to verb
    * Default morphology = SING, IND, PRES, 3
  * ARTICLE changes the part-of-speech to article
    * Default morphology = SING, MASC, DEF
  * ADJ changes the part-of-speech to adjective
    * Default morphology = SING, MASC
  * ADV changes the part-of-speech to adverb
    * Default morphology = SING, MASC
* GENDER (GÉNERO)
  * MASC - makes the word masculine (if possible)
    * e.g. apuesta + `<MUTATE type=GENDER-MASC/>` = apuesto
  * FEM - makes the word feminine (if possible)
    * e.g. harto + `<MUTATE type=GENDER-MASC/>` = harta
* NUMBER (NÚMERO)
  * SING - makes the word singular
  * PLU - makes the word plural
* DEFINITE
  * DEF - makes the word definite
  * IND - makes the word indefinite
* CASE
  * NOM - makes the word nominative
  * ACC - makes the word accusative
  * DAT - makes the word dative
* PRONOUN_TYPE (for personal pronouns only)
  * BASE - makes the pronoun of the normal base type (e.g. *él*, *yo*, *usted*)
  * CLITIC - makes the pronoun of the clitic type (e.g. *le*, *se*, te*, *me*)
* REFLEXIVE (for personal pronouns only and only affects clitic pronouns)
  * YES - makes the pronoun reflexive (e.g. *se*, *me*, *te*)
  * NO - makes the pronoun non-reflexive (e.g. *le*, *se*, te*, *me*)
* PERSON (PERSONA)
  * 1 - makes the word first-person (must be verb)
    * e.g. son + `<MUTATE type=PERSON-1` = somos
  * 2 - makes the word second-person (must be verb)
  * 3 - makes the word third-person (must be verb)
* MOOD (MODO)
  * IND (INDICATIVO) - makes the word indicative in mood (must be verb)
  * SUB (SUBJUNTIVO) - makes the word subjunctive in mood (must be verb)
  * **DEPRECATED**[^1] PROG (PROGRESIVO) - makes the word progressive in mood (must be verb) (adds the correct conjugation of *estar* before the verb)
  * **DEPRECATED**[^1] PERF (PERFECTO) - makes the word perfect in mood (must be verb) (adds the correct conjugation of *haber* before the verb)
  * **DEPRECATED**[^1] PERF-SUBJ (PERFECTO SUBJUNCTIVO) - makes the word perfect subjunctive in mood (must be verb) (adds the correct conjugation of *haber* before the verb)
  * GER (PARTICIPIO PRESENTE) - makes the word in the gerund form, distinct from progressive which also adds the correct conjugation of *estar* (must be verb)
  * PAST-PART (PARTICIPIO PASADO) - makes the word in the past participle form (must be verb)
  * INF (INFINITIVO) - makes word in the infinitive form (i.e. no mood) (must be verb)
* TIME (TIEMPO)
  * PRES (PRESENTE)
  * PRET (PRETÉTERITO)
  * IMP (IMPERFECTO)
  * CND[^2] (CONDICIONAL)
  * FUT (FUTURO)

  [^1]: Mutations which require adding extra words are deprecated. The functionality remains in decode.py, but these types of mutations will not result from automatically generated labels from generate.py or label.py.
  [^2]: Note that the conditional is considered a tense. There is some disagreement over whether the conditional is a mood or a tense in Spanish, but I consider it a tense in my label schema.

## Token-Label Stacking
You may have noticed that these labels are not mutually exclusive. Some labels are incompatible or redundant together, but many would be expected to operate in conjunction. Therefore, the textual label is formatted with tabs in between each token's label. If there is no tab between two labels, then all of those labels apply to the same token. For example:

```
espero que corre tú bien.
<MUTATE type=CAPITALIZE-TRUE/>  <KEEP/>  <MUTATE type=PERSON-2/><MUTATE type=MOOD-SUBJ/><ADD token=tú/> <DELETE/>  <KEEP/>  <KEEP/>
```

changes the sentence to:

`Espero que tú corras bien.`

This means that this defines a *multi-label classification task* for each token. The expected output for each token is an integer vector with the length of the number of possible labels (30). Each label dimension is binary **except** ADD (value = vocab index), COPY (value = token index), and REPLACE (value = vocab index).

Note this is still a WIP, so I'm happy to take feedback for adjustments of this formatting. This is my first iteration and I do not yet have performance evaluations with this task design.