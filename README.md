# synth_gec_es
[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-yellow.svg)](README-es.md)

SYNTHetic Grammatical Error Correction for Spanish (ES) is a system for generating synthetic GEC data for common Spanish grammatical errors to train a GEC model.

Generally to be used to augment smaller high-quality training sets as in *GECToR – Grammatical Error Correction: Tag, Not Rewrite* (2020)

**WORK IN PROGRESS**

## Major TODOs

These are just for me to record what needs to be done, especially as many of these problems are absolutely not trivial

* First priority is completing the morphology dictionary, haven't been able to find anything exactly like what I want besides maybe scraping WordReference or Spanish Dictionary. Essentially, my goal is to to create a surface realization model for Spanish w/ a dictionary
  * I'd like to avoid needing to scrape this entire thing myself, so I am leaning on existing dump parses of resources like Wikitionary
  * Noun/pronoun/article/adjective/adverb morphology is *mostly* solvable with parsed Wikitionary dump, /tatuylonen/wiktextract has kindly already done this
  * Wikitionary is missing mood and has inconsistent specificity of verb categories
  * Verb categories (most complex) can most likely be generated from /ghidinelli/fred-jehle-spanish-verbs/
  * Therefore, the data is there, just need to reconcile these two sources
* Second priority is implementing decoding logic (incorrect sentence + labels -> corrected sentence)
  * Getting this working correctly will allow finalization of label schema (at least for v1)
  * Current WIP features for this:
    * Determine *current* morphosyntactic categories, requirement for mutate to work correctly (e.g. need to be able to change mood while maintaing current time, person, and number)
      * ~requires disambiguation based on POS tag and dependent noun (offloading most disambiguation logic onto existing dependency tree models), placeholder logic just picks first exact match~ spaCy has a built-in morphology analysis that works great for Spanish, **solved problem** 
        * disambiguation needs to be able to correct itself if the mutation is incompatible with the current categories (e.g. if it thought *coma* was a singular noun, but then receives a `<MUTATE param=MOOD-IND/>`, it should understand its initial determination was incorrect and try again with the knowledge that the POS tag is actually VERB (this is an especially complicated case because we would also need to correct the lemma (*coma* vs. *comer*), an exceptional case where both the lemmatizer and POS tagger failed)
        * ~need default hierarchy for categories if certain categories are impossible to resolve (e.g. *que trabaje mejor* could be *que (yo) trabaje mejor* or *que (él) trabaje mejor*)~ **solved by spaCy**, but will have a fallback default hierarchy for each POS just in case
    * Need to finalize how to handle clitic pronouns (e.g. *te amo* and *cómetelas*), right now they are treated as separate tokens to simplify mutate logic
      * Probably should not be treated as separate tokens in final version because reversal is required for edge case POS-IMP -> NEG-IMP and vice versa (e.g. *no te muevas* vs. *muévate*). However, this transformation is still possible because we can just use DELETE + COPY to reverse the position of each pronoun. I'd like to simplify this as it is a common error, but at some point I may need to stop diving too deep into creating error-specific logic. Almost feels like I'm accidentally trying to make a rule-based GEC system just to solve this synthetic data problem.
      * Overall, implementing implicit reversal logic seems like a bad call. I may be thinking too much about this specific syntactic transformation because clitic pronouns in imperative sentences in Spanish has been a research interest of mine
  * Unsolved:
    * How to handle orthographic errors, currently solved by replace (expensive, to be avoided), but should be cheap, maybe add a cheaper operation than replace (but more expensive than mutate) if edit distance between mispelling and correction is below a threshold (same actual calculation cost in a model as it requires an index the size of the vocabulary, but could be more useful in explanation generation)
* Finish tests (**test-driven development brain worms**)
* Design and implement labelling algorithm for errorful + correct pairs. Should be able to adapt minimum edit distance + backtrack to my transformations, but the specifics currently vex me. Solving this algorithm in an efficient manner is why I removed the MOVE transformation in favor of a reduced cost COPY + DELETE.
* Design some form of error definition config format for generate.py. I could hard-code error types, but I think it makes more sense to have an easy way to add error types if someone (including me) realizes a type of error is missing.
  * Currently, the error types I can think of that need to be generated are subject-verb disagreement, gender disagreement, incorrect case (e.g. *tú* vs. *ti*), common word usage errors (e.g. *bien* vs. *bueno*, *por* vs. *para*), grammatically impossible mood (e.g. *\*ojalá se rinde* has the wrong mood), erroneous inclusion of a word (especially preposition or article), missing word (especially preposition or article), orthographic error, and capitalizaition error
    * Ugh, but what about cases where the generation of the error isn't actually an error (e.g. *soy feliz* and *estoy feliz* are both grammatically fine, a correction requires knowledge of intent (are they a happy person or just feeling happy?))
    * This system is *defining* the patterns of grammatical errors, then translating it to a format a neural model can understand and extrapolate from. This is why this synthetic data should *augment* an existing GEC training dataset which will contain missing errors not found in this synthetic data. Flexibility of the model will improve the more errors are included here

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
* `<PRE-COPY param=i/>`
  * Add the given token (based on token index in the sentence) immediately before this token
* `<POST-COPY param=i/>`
  * Add the given token (based on token index in the sentence) immediately after this token
* `<MUTATE param=i>`
  * Mutate the morphology of this token based on a given type. Major types include CAPITALIZE, GENDER, NUMBER, PERSON, MOOD, and TIME. Further discussion [here](#mutation-types)
* `<REPLACE param=i/>`
  * Replace this token with the given token (based on token index in vocab.txt)

### Why Both PRE- and POST-ADD/COPY?
PRE-ADD/COPY and POST-ADD/COPY are redundant. In reality, all sentence transformations are obviously possible with only one of these. However, I believe having both possible transformations may allow for more flexibility in error correction. My hypothesis is that allowing a token to be tagged w/ either ADD type will train a more bidirectional understanding of how to build phrases.

For example, lets say we have the incorrect sentence:

  (1) \*Él es trabajador muy.

The most natural way to correct this is:

```
<KEEP/> <KEEP/> <PRE-COPY param=3/> <DELETE/> <KEEP/>
```

which results in:

  (2) Él es muy trabajador.

This is because *muy* is in the same hierarchical grouping as *trabajador*. *muy* modifies *trabajador*.

If we only had POST-ADD (i.e. APPEND), the correction would be:

```
<KEEP/> <POST-COPY param=3/> <KEEP/> <DELETE/> <KEEP/>
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

This means that this defines a *multi-label classification task* for each token. The expected output for each token is an integer vector with the length of the number of possible labels (30). Each label dimension is binary **except** PRE-ADD/POST-ADD (value = token index), PRE-COPY/POST-COPY, and REPLACE (value = token index).

Note this is still a WIP, so I'm happy to take feedback for adjustments of this formatting. This is my first iteration and I do not yet have performance evaluations with this task design.

  
