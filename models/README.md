# Baseline Model Fine-Tuning and Evaluation

To establish baseline performance metrics of models using this synthetic code, there are four models defined in this repo:
* Token-level classification using BETO (Spanish BERT) for sequence classification with only the existing COWS-L2H corpus as training data
* Token-level classification using BETO with COWS-L2H and my synthetically-generated corpus as training data
* Seq2seq classification (as in NMT-inspired models) using BARTO (Spanish BART) with the existing COWS-L2H corpus as training data
* Seq2seq classification (as in NMT-inspired models) using BARTO with COWS-L2H and my synthetically-generated corpus as training data

Therefore, this is a 2x2 experimental design:

| Model | COWS-L2H | COWS-L2H + Synthetic Dataset |
| ---- | ---- | ----------- |
| BETO (Token classification) | X | X |
| BARTO (Seq2seq) | X | X |

## Running the models

**There is a demo of all four models. Run model_demo.py to try them out (in a very bare-bones text demo).**

The seq2seq models can be fine-tuned using the seq2seq_model.py script (--synth argument to use synthetic data).

The token-level classification models can be fine-tuned using the token_model.py script (--synth argument to use synthetic data). **IMPORTANT**: The first time this model is partially initialized from the BETO checkpoint, the weights don't load for some reason, so a large warning prints. This is ok because I overwrite the encoder after initialization. Once it has been trained the first time, there is no warning because the weights are correctly initialized. I don't know why this happens.

However, I have already fine-tuned the models for GEC, so fine-tuning yourself is only for replication. The fine-tuned models are available on HuggingFace at:

* BETO (COWS-L2H only): SkitCon/gec-spanish-BARTO-COWS-L2H
* BETO (COWS-L2H + Synthetic): SkitCon/gec-spanish-BARTO-SYNTHETIC
* BARTO (COWS-L2H only): SkitCon/gec-spanish-BETO-TOKEN-COWS-L2H
* BARTO (COWS-L2H + Synthetic): SkitCon/gec-spanish-BETO-TOKEN-SYNTHETIC

Instructions for running each model are on the HuggingFace model card for each model, but you can run them from the demo.

Note that to use the BETO token-level models, you need the utils, models, and lang_def directories for importing the decoding functions.

Evaluation with some error analysis can be ran with (note this will use the HuggingFace models):

```
cd models
python3 eval_models.py
```

## Results

Note that these results are without any hyperparameter tuning for the specific task. All of these use the same basic training strategy: mini-batch gradient descent, Adam optimizer, batch size 16, 8 epochs. Everything else is default for the respective base models.

### BLEU Score

I use BLEU score between the predicted sentence and target sentence for my main measurement of performance between models. I use this because the seq2seq models do not have token labels to score accuracy or F1 on, so a comparison with those metrics would be unfair.

On held-out test data (n=2,000 sentences) from COWS-L2H:
* **BASELINE** Seq2Seq (COWS-L2H only): 0.846 BLEU
* Seq2Seq (COWS-L2H + Synthetic): 0.851 BLEU (+0.005 compared to baseline)
* Token Classification (COWS-L2H only): 0.735 BLEU (-0.111 compared to baseline)
* Token Classification (COWS-L2H + Synthetic): 0.745 BLEU (-0.101 compared to baseline)

In both the seq2seq models and token classification models, my synthetic data improves performance on the test data. I believe with targeted additions of certain errors to the generation code the synthetic data could improve performance even more. However, the token-classification models (which I thought would perform better) actually perform much worse than the baseline. Only the seq2seq model with synthetic data performs better than the baseline. I believe the reason for this is that the model fails to converge correctly on the transformation parameters (e.g. vocab index of the token to replace, mutation type, etc.). Despite my hypothesis, the token classification model using my current design takes *more* data to train, not less.

### Targeted Tests

As a tangible comparison between models, I have constructed a series of basic sentence corrections which the models *should* be able to do if they were to be used in practice. The sentences are:

(1) \*yo va al tienda. -> (Yo) voy a la tienda. or Va a la tienda. (Incorrect capitalization, basic subject-verb disagreement, and gender disagreement)

(2) \*Gracias para invitarme. -> Gracias por invitarme (Very simple mixup of *por* and *para*)

(3) \*Espero que tú ganas el juego. -> Espero que (tú) ganes el juego. (Redundant pronoun (ok if not deleted) and incorrect mood (subjunctive is required here))

(4) \*Soy una mujer y soy bello. -> Soy una mujer y soy bella. (Contextual gender mismatch)

(5) \*Le dije que estaba avergonzada, pero me llamó emotivo. -> Le dije que estaba avergonzada, pero me llamó emotiva. (More complicated gender mismatch)

I also included these sentences to see if the model has internalized gender bias.

(6) Mi novia me dijo que me parecía bella esta noche. (There should be no correction here, but I wonder if the model will assume the speaker is a man because they have a girlfriend and correct *apuesta* to *apuesto*)

(7) Trabajo como albañil así que siempre me siento cansada. (There should be no correction here, but I wonder if the model will assume the speaker is a man because they work as a bricklayer and correct *cansada* to *cansado*)


The results of sentences 1-7:
* **BASELINE** Seq2Seq (COWS-L2H only): 5/7 pass
* Seq2Seq (COWS-L2H + Synthetic): 5/7 pass
* Token Classification (COWS-L2H only): 3/7 pass
* Token Classification (COWS-L2H + Synthetic): 4/7 pass

Detailed descriptions of the errors are in the [error analysis section](#error-analysis).

## Error analysis

The most obvious common issue is with the token models. While the model is good at correcting simple errors, the model fails to converge on determining the correct replacement token for a REPLACE transformation. This results in most REPLACE transformations having the parameter 0 (the index of \[MASK\]). For example, the erroneous sentence *\*Él es en el parque.* has a grammatical error where the word *es* is used instead of *está*. The model correctly detects that *es* should be replaced, but instead of choosing *está* as the replacement, it chooses \[MASK\]. I believe this could be solved with a supplemental pre-training procedure (maybe pre-train the model to choose a replacement?) or using a different model (maybe use BARTO for the token model too to improve decoder performance).

For further analysis beyond BLEU score, I analyzed the ability of the models to correctly detect if an error is or is not present based on the output. Therefore, each sentence is put into one of five categories:

* TP: True positive, model correctly identifies the sentence as having an error AND output sentence matches the gold sentence
* FP: False positive, model incorrectly identifies the sentence as having an error when there is no error present
* TN: True negative, model correctly identifies the sentence as having no error and makes no changes to the sentence
* FN: False negative, model incorrectly identifies the sentence as having no error when there are actually errors in the sentence
* Incomplete correction: The model correctly identifies the sentence as having an error, but the output does not match the gold sentence.

| | s2s (COWS-L2H only) | s2s (COWS-L2H + Synthetic) | token (COWS-L2H only) | token (COWS-L2H + Synthetic) |
| ---- | ---- | --- | --- | --- |
| TP | 517 | 532 | 179 | 194 |
| FP | 76 | 66 | 102 | 99 |
| TN | 599 | 609 | 558 | 561 |
| FN | 239 | 241 | 261 | 287 |
| Incomplete correction | 569 | 552 | 900 | 859 |

Some key takeways are:
* All models have very few false positives. The models do very well at not correcting a sentence that is grammatically fine.
* The categories with the most divergence between model architectures are true positives and incomplete corrections. The token models do *much* worse at correctly correcting sentences, even if they detect errors at around the same rate.

With the sentences split based on whether there are errors or not, we can get the BLEU score for each type of sentence:

| | s2s (COWS-L2H only) | s2s (COWS-L2H + Synthetic) | token (COWS-L2H only) | token (COWS-L2H + Synthetic) |
| ---- | ---- | --- | --- | --- |
| BLEU ONLY on sentences with no errors | 0.964 | 0.978 | 0.954 | 0.955 |
| BLEU ONLY on sentences with errors | 0.794 | 0.799 | 0.645 | 0.660 |

This analysis supports the same conclusion as the category counts. The token models do much worse at correctly changing a sentence. We can see this in detail by looking at the results of the selected sentences test set (different tokens bolded):

* Seq2seq (COWS-L2H only)
  * Sentence 3 fails:
    * Predicted sentence: Espero que tú **ganas** el juego.
    * Target: Espero que tú **ganes** el juego.
    * Explanation: Model fails to detect that *ganas* should be in the subjunctive mood.
  * Sentence 5 fails:
    * Predicted sentence: Le dije que estaba avergonzada, pero me llamó **emotivo**.
    * Target: Le dije que estaba avergonzada, pero me llamó **emotiva**.
    * Explanation: Model fails to infer that *emotivo* refers to the speaker who is already identified as female by the use of *avergonzada*, so *emotivo* should be *emotiva* (feminine version).
* Seq2seq (COWS-L2H + Synthetic)
  * Sentence 3 fails:
    * Predicted sentence: Espero que tú **ganas** el juego.
    * Target: Espero que tú **ganes** el juego.
    * Explanation: Model fails to detect that *ganas* should be in the subjunctive mood.
  * Sentence 5 fails:
    * Predicted sentence: Le dije que estaba avergonzada, pero me llamó **emotivo**.
    * Target: Le dije que estaba avergonzada, pero me llamó **emotiva**.
    * Explanation: Model fails to infer that *emotivo* refers to the speaker who is already identified as female by the use of *avergonzada*, so *emotivo* should be *emotiva* (feminine version).
* Token (COWS-L2H only)
  * Sentence 1 fails:
    * Predicted sentence: Va **\[MASK\]** tienda.
    * Target: Va **a la** tienda.
    * Explanation: Model correctly deletes pronoun *yo* to correct subject-verb disagreement and correctly identifies that *al* is the incorrect preposition + article for *tienda*, but it fails to identify the correct replacement (*a la*) and defaults to the first word in the dictionary, \[MASK\].
  * Sentence 2 fails:
    * Predicted sentence: Gracias **para** invitarme.
    * Target: Gracias **por** invitarme.
    * Explanation: Model makes no change to the sentence. It fails to identify that *para* is the correct preposition and *por* should be used.
  * Sentence 3 fails:
    * Predicted sentence: Espero que **Ganas** el juego.
    * Target: Espero que **ganes** el juego.
    * Explanation: Model correctly deletes redundant pronoun *tú* (this is more of a stylistic improvement) and correctly identifies that *ganas* should be changed, but does not correctly identify that *ganas* should be in the subjunctive mood. Instead, the model capitalizes the word. It is worth noting that the parameter for mutate that signifies capitalizing the word is 0 in scalar form, so the capitalization is simply a case of choosing the mutation type with the index 0 (same type of error as choosing \[MASK\] for replace).
  * Sentence 5 fails:
    * Predicted sentence: Le dije que estaba avergonzada, pero me llamó emo \[MASK\].
    * Target: Le dije que estaba avergonzada, pero me llamó **emotiva**.
    * Explanation: Model actually correctly identifies that *emotivo* should change (though in this case the subword *##tivo* due to how it was tokenized), but does not correctly update gender. Instead, \[MASK\] is chosen again.
* Token (COWS-L2H + Synthetic)
  * Sentence 1 fails:
    * Predicted sentence: voy a **yo** tienda.
    * Target: Voy a **la** tienda.
    * Explanation: Model correctly deletes redundant *yo* pronoun and corrects *va* to the first person (*voy*). The model also correctly tries to fix *al* by changing it to *a* + something, but erroneously copies the first word *yo* instead of replacing it with *la*. The model also fails to capitalize the first word.
  * Sentence 3 fails:
    * Predicted sentence: Espero que **Ganas** el juego.
    * Target: Espero que **ganes** el juego.
    * Explanation: Model correctly deletes redundant pronoun *tú* (this is more of a stylistic improvement) and correctly identifies that *ganas* should be changed, but does not correctly identify that *ganas* should be in the subjunctive mood. Instead, the model capitalizes the word. It is worth noting that the parameter for mutate that signifies capitalizing the word is 0 in scalar form, so the capitalization is simply a case of choosing the mutation type with the index 0 (same type of error as choosing \[MASK\] for replace).
  * Sentence 5 fails:
    * Predicted sentence: Le dije que estaba avergonzada, pero me llamó emo \[MASK\].
    * Target: Le dije que estaba avergonzada, pero me llamó **emotiva**.
    * Explanation: Model actually correctly identifies that *emotivo* should change (though in this case the subword *##tivo* due to how it was tokenized), but does not correctly update gender. Instead, \[MASK\] is chosen again.

Notably, no model correctly fixes sentences 3 and 5. Additionally, the token model without the synthetic data is the only model to fail sentence 2. My takeaway from this is that none of the models are necessarily ready for real world use without supervision. Sentence 5 is a difficult correction (requires inferred gender), so I think it is ok if a model doesn't catch that error. However, I am surprised sentence 3 fails for all models. This is a textbook example of where subjunctive mood is required, so something similar to this is almost guaranteed to be in the training data. This kind of error *should* be easy for the models.

Additionally, none of the sentences fail the test for gender bias, but this may be because the models are biased toward *not* correcting a sentence.