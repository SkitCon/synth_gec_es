# synth_gec_es
[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-yellow.svg)](README-es.md)

**Scripts para replicar el fine-tuning de los models bases es en [./models](models/README.md).**

**versión 0.7.5**

La corrección de errores gramaticales sintéticos para ESpañol es un sistema para generar datos GEC syntéticos para errors gramaticales comúnes en español para entrenar un modelo GEC.

Generalmente se usa para amplificar conjuntos de datos más pequeños de alta calidad igual que en *GECToR – Grammatical Error Correction: Tag, Not Rewrite* (2020)

**TRABAJO EN PROCESO**
Todos los scripts y modelos funcionan, pero aún queda posible que hay errores menores en casos extremos. En general, si falla un script, un mensaje imprimirá y el script seguirá con el resto del archivo, pasanda esa frase.

Se requiren estas librerías para todos los scripts:
```
bs4 == 0.0.2
spacy == 3.7.6
unidecode == 1.3
transformers == 4.31.0
torch==2.4.1
numpy = 1.24.4
nltk == 3.8.1
scikit-learn==0.24.2

# Ejecuta para descargar el modelo de spaCy
python -m spacy download es_dep_news_trf
```

## Índice de Materias
* [Resumen](#resumen)
* [Scripts](#scripts)
  * [generate.py](#generate.py)
  * [decode.py](#decode.py)
  * [label.py](#label.py)
* [Definiciones](#definiciones)
* [Tipos de Mutación](#tipos-de-mutación)
* [Etiquetas Múltiples](#etiquetas-múltiples)
* [Generar el Diccionario Morfológico](#generar-el-diccionario-morfologico)
* [Limitaciones](#limitaciones)

## Resumen

Este repo contiene un conjunto de scripts de herramientas para:
1. **Generar** errores sintéticos en frases correctas en español usando [generate.py](#generatepy). Se genera errores sintéticos con definiciones basadas en reglas definidas como archivos JSON. Diseñé este script a ser modular para permitir la adición de nuevos errores facilmente o cambiar los pesos de errores. Este script también puede generar transformaciones de token para cada par de frases generado por este script. Se define las transformaciones [de abajo](#definiciones).
2. **Decodificar** frases con errores + transformaciones de token a frases corregidas usando [decode.py](#decodepy). Generalmente, se usa este script para decodificar un archivo de saldio de un modelo fine-tuned que se entrenó en estas transformaciones de token.
3. **Etiquetar** frases con errores + frases correctas/corregidas con transformaciones de token usando [label.py](#labelpy). Usa este script si ya tienes datos de entrenamiento para corrección de errores gramaticales, pero quieres entrenar tu modelo para clasificar por token en vez de seq2seq.
4. Funciones de utilidad para incorporar facilemente este esquema en un modelo en [utils.py](utils/utils.py), por ejemplo para convertir un string de transformaciones de token a un vector (y viceversa), decodificar una sola frase, y resolución de morfología (e.g. *estar* + MOOD-SUB + PERSON-2 + TIME-PRES = *estes*).

También, hay cuatro models fine-tuned disponsble que usa datos sintéticos generados por estos scripts. Instrucciones para usar estos models se ubica en [./models](models/README.md).

## Scripts

Nota: Si usas la misma organización de archivos como en el repo (por ejemplo si clonaste este repo), no te preocupes con los argumentos de --dict_file, --vocab_file, --error_files, --spacy_model, o --tokenizer_model.

### generate.py

El script principal es generate.py. Se ejecuta como:

```
python generate.py INPUT_FILE OUTPUT_FILE [--error_files] [ERROR_FILE_1 ERROR_FILE_2 ... ERROR_FILE_N] [-min/-min_error] [minimum number of errors in a sentence] [-max/--max_error] [maximum number of errors in a sentence] [-d/--dict_file] [dictionary file] [--vocab_file] [vocab file] [--spacy_model] [spaCy model name] [--tokenizer_model] [tokenizer model path] [--seed] [seed] [-n/--num-sentences] [number of sentences to generate for each original] [--n_cores] [number of cores to use] [-t/--token] [--verify] [-v/--verbose] [-sw/--silence_warnings] [--strict]
```

Este script se usa para generar frases con errores de un archivo de corpus con frases sin errores.

* input file es una ruta de entrada a un archivo .txt con una frase por cada línea. Tenga en cuenta que este script espera datos sin etiqueta.
* output file es opcional, define la ruta de salida para guardar los datos sintéticos. Por omisión, la ruta de salida es \[input file name\]_synth.txt.
* --error_files es una lista de archivos JSON con errores definidos
* --min_error es número mínimo de errores de generar en una frase
* --max_error es número máximo de errores de generar en una frase
* --dict_file es la ruta al archivo JSON que define el diccionario morfológico
* --vocab_file es la ruta al archivo vocab.txt del modelo
* --spacy_model es el modelo de spaCy que se usará para analizar morfología
* --tokenizer_model es el modelo de usar para tokenización (ruta local o HuggingFace)
* --seed es la semilla aleatoria
* --num-sentences es el número de frases con errores que se generará de cada frase original en el corpus de entrada. Por omisión es 1.
* --n_cores especifica el número de núcleos disponible para el script. Si es 1, no se usa multi-processing.
* --token significa que el archivo de saldio incluirá transformaciónes de token (leer más [de abajo](#definiciones))
* --verify significa que las transformaciones de token generadas por el script se verificará usando el algoritmo de decodifiar para asegurar que el resultado iguala la frase correcta. No hay efecto si no se usa --token. Nota que generalmente esta opción aumenta el tiempo de procesimiento por dos, pero garantiza que las transformaciones generados son válidas. Actualmente, recomiendo esta opción.
* --verbose significa que el script imprimirá mensajes de depuración, NO SE RECOMIENDA CON MULTI-PROCESSING.
* --silence_warnings significa que se silenciará mensajes de advertencia, por ejemplo si hay una mutación (MUTATE) que falla y que se necesita reemplazarla con una reemplazación (REPLACE).
* --strict significa que el script no incluirá frases en que una mutación (MUTATE) falle en el archivo de salida.

### decode.py

Se ejecuta como:

```
python3 decode.py INPUT_FILE [OUTPUT_FILE] [-d/--dict_file] [dictionary file] [--vocab_file] [vocab file] [--spacy_model] [spaCy model name] [--tokenizer_model] [tokenizer model path] [--n_cores] [number of cores to use] [-v/--verbose] [-sw/--silence_warnings]
```

Este script se usa un archivo con frases con errores + etiquetas de token para convertirlas a frases corregidas.

* input file es una ruta de entrada a un archivo .txt con una frase por cada línea. Tenga en cuenta que este script espera datos sin etiqueta.
* output file es opcional, define la ruta de salida para guardar los datos sintéticos. Por omisión, la ruta de salida es \[input file name\]_decoded.txt.
* --dict_file es la ruta al archivo JSON que define el diccionario morfológico
* --vocab_file es la ruta al archivo vocab.txt del modelo
* --spacy_model es el modelo de spaCy que se usará para analizar morfología
* --tokenizer_model es el modelo de usar para tokenización (ruta local o HuggingFace)
* --n_cores especifica el número de núcleos disponible para el script. Si es 1, no se usa multi-processing.
* --verbose significa que el script imprimirá mensajes de depuración, NO SE RECOMIENDA CON MULTI-PROCESSING
* --silence_warnings significa que se silenciará mensajes de advertencia, por ejemplo si hay una mutación (MUTATE) que falla y que se necesita reemplazarla con una reemplazación (REPLACE).

### label.py

Se ejecuta como:

```
python3 label.py INPUT_FILE [OUTPUT_FILE] [-d/--dictionary-file] [dictionary file] [--vocab_file] [vocab file] [--spacy_model] [spaCy model name] [--tokenizer_model] [tokenizer model path] [--n_cores] [number of cores to use] [--verify] [-v/--verbose] [-sw/--silence_warnings] [--strict]
```

Este script se usa frases con errores + frases sin errores y crea etiquetas de token para ellas.

* input file es una ruta de entrada a un archivo .txt con una frase por cada línea. Tenga en cuenta que este script espera datos sin etiqueta.
* output file es opcional, define la ruta de salida para guardar los datos sintéticos. Por omisión, la ruta de salida es \[input file name\]_labeled.txt.
* --dict_file es la ruta al archivo JSON que define el diccionario morfológico
* --vocab_file es la ruta al archivo vocab.txt del modelo
* --spacy_model es el modelo de spaCy que se usará para analizar morfología
* --tokenizer_model es el modelo de usar para tokenización (ruta local o HuggingFace)
* --n_cores especifica el número de núcleos disponible para el script. Si es 1, no se usa multi-processing.
* --verify significa que las transformaciones de token generadas por el script se verificará usando el algoritmo de decodifiar para asegurar que el resultado iguala la frase correcta. No hay efecto si no se usa --token. Nota que generalmente esta opción aumenta el tiempo de procesimiento por dos, pero garantiza que las transformaciones generados son válidas. Actualmente, recomiendo esta opción.
* --verbose significa que el script imprimirá mensajes de depuración, NO SE RECOMIENDA CON MULTI-PROCESSING
* --silence_warnings significa que se silenciará mensajes de advertencia, por ejemplo si hay una mutación (MUTATE) que falla y que se necesita reemplazarla con una reemplazación (REPLACE).
 * --strict significa que el script no incluirá frases en que una mutación (MUTATE) falle en el archivo de salida.

## Definiciones

NOTA: Para acomodar la añadición de palabras al fin de una frase (porque ADD prepone una palabra), se añade \[EOS\] al final de frases y estos tokens reciben transformaciones de token. El código de generación ya hace esta operación, así que no se necesita hacerlo. Sin embargo, al entrenar un model, se necesita asegurar que el token de \[EOS\] existe y que pasa por el nivel de clasificación.

Las transformaciones de token principales son:
* `<KEEP/>`
  * No cambiar el token
* `<DELETE/>`
  * Borrar el token
* `<ADD param=i/>`
  * Añadir el token *i* (por índice de token en vocab.txt) justo antes de este token
* `<COPY-REPLACE param=i/>`
  * Reemplazar el token con el token *i* (por índice de token en la frase)
* `<COPY-ADD param=i/>`
  * Añadir el token *i* (por índice de token en la frase) justo antes de este token
* `<MUTATE param=tipo de mutación]>`
  * Mutar la morfología de este token basado en el param. Los tipos principales son CAPITALIZE, GENDER, NUMBER, PERSON, MOOD, y TIME. Más información [aquí](#tipos-de-mutación)
* `<REPLACE param=i/>`
  * Reemplazar este token con el token *i* (por índice de token en vocab.txt)

## Tipos de Mutación

* CAPITALIZE (ESCRIBIR CON MAYÚSCULAS)
  * TRUE - escribir la palabra con mayúsculas
  * FALSE - escribir la palabra con minúsculas
* POS (PARTE DE ORACIÓN)
  * Nota: Porque la morfología en partes de oración diferentes generalmente no son compatibles, hay morfología predeterminada definida por cada parte de oración
  * NOUN - convertir la parte de oración a sustantivo
    * Morfología predeterminada = SING, MASC
  * PRONOUN - convertir la parte de oración a pronombre
    * Morfología predeterminada = SING, MASC, NOM
  * PERSONAL_PRONOUN - convertir la parte de oración a pronombre personal
    * Morfología predeterminada = SING, MASC, NOM, BASE, NO
  * VERB - convertir la parte de oración a verbo
    * Morfología predeterminada = SING, IND, PRES, 3
  * ARTICLE - convertir la parte de oración a artículo
    * Morfología predeterminada = SING, MASC, DEF
  * ADJ - convertir la parte de oración a adjetivo
    * Morfología predeterminada = SING, MASC
  * ADV - convertir la parte de oración a adverbio
    * Morfología predeterminada = SING, MASC
* GENDER (GÉNERO)
  * MASC - convertir la palabra al masculino
    * e.g. apuesta + `<MUTATE type=GENDER-MASC/>` = apuesto
  * FEM - convertir la palabra al feminino
    * e.g. harto + `<MUTATE type=GENDER-MASC/>` = harta
* NUMBER (NÚMERO)
  * SING - convertir la palabra al singular
  * PLU - convertir la palabra al plural
* DEFINITE
  * DEF - convertir la palabra al definido
  * IND - convertir la palabra al indefinido
* CASE
  * NOM - convertir la palabra al nominativo
  * ACC - convertir la palabra al acusativo
  * DAT - convertir la palabra al dativo
* PRONOUN_TYPE (solo para pronombres personales)
  * BASE - convertir la palabra a la forma base del pronombre personal (e.g. *él*, *yo*, *usted*)
  * CLITIC - convertir la palabra a la forma clítica del pronombre personal (e.g. *le*, *se*, te*, *me*)
* REFLEXIVE (solo para pronombres personales y solo afecta pronombres clíticos)
  * YES - convertir la palabra a la forma reflexiva (e.g. *se*, *me*, *te*)
  * NO - convertir la palabra a la forma normal (e.g. *le*, *se*, te*, *me*)
* PERSON (PERSONA)
  * 1 - convertir la palabra a la primera persona
    * e.g. son + `<MUTATE type=PERSON-1` = somos
  * 2 - convertir la palabra a la segunda persona
  * 3 - convertir la palabra a la tercera persona
* MOOD (MODO)
  * IND (INDICATIVO) - convertir la palabra al indicativo
  * SUB (SUBJUNTIVO) - convertir la palabra al subjuntivo
  * **DEPRECATED**[^1] PROG (PROGRESIVO) - convertir la palabra al progresivo (con la conjugación correcta de *estar* antes del verbo)
  * **DEPRECATED**[^1] PERF (PERFECTO) - convertir la palabra al perfecto (con la conjugación correcta de *haber* antes del verbo)
  * **DEPRECATED**[^1] PERF-SUBJ (PERFECTO SUBJUNCTIVO) - convertir la palabra al perfecto subjuntivo (con la conjugación correcta de *haber* antes del verbo)
  * GER (PARTICIPIO PRESENTE) - convertir la palabra al participio presente, distinto del progresivo que también añade la conjugación correcta *estar*
  * PAST-PART (PARTICIPIO PASADO) - convertir la palabra al participio pasado
  * INF (INFINITIVO) - convertir la palabra al infinitivo (i.e. sin modo)
* TIME (TIEMPO)
  * PRES (PRESENTE)
  * PRET (PRETÉTERITO)
  * IMP (IMPERFECTO)
  * CND[^2] (CONDICIONAL)
  * FUT (FUTURO)

  [^1]: Mutaciones que require la añadición de palabas extras son deprecadas. Aun queda la funcionalidad en decode.py, pero estos tipos de mutaciones no se generará por generate.py o label.py.
  [^2]: Nota que el condicional se define como tiempo. Hay discursión si el condicional es un modo o un tiempo en español, pero yo lo defino como tiempo en este esquema de transformaciones.

## Etiquetas Múltiples
Es posible que hayas dado cuenta de que estas etiqueta no son mutuamente excluyentes. Algunos de las etiqueta no son compatibles o que en conjunto resultan redundantes, pero muchas se esperan operar en conjunto. Por los tanto, la etiquetas se separa con pestañas. Si no hay una pestañas entre dos etiquetas, estas etiqueta aplica al mismo token. Por ejemplo:

```
espero que corre tú bien.
<MUTATE param="CAPITALIZE-TRUE"/>  <KEEP/>  <MUTATE param="PERSON-2"/> <MUTATE param="MOOD-SUBJ"/> <COPY-ADD param=3/> <DELETE/>  <KEEP/>  <KEEP/>
```

convierte la frasa al:

`Espero que tú corras bien.`

Significa que el esquema define clasificación multietiqueta para cada token. La salida esperada para cada token es un vector de integer con el tamaño de max_labels * 2. En el vector de salida, cada transformación se define por un par de integers en que lo primero define el tipo de transformación (e.g. KEEP, REPLACE, MUTATE) y el segundo define el parámetro (e.g. MOOD-SUB, índice 1, índice 1,243). Las transformaciones de token del vector se aplican del comienzo al final del vector.

Por ejemplo, definimos esta transformación de token:

Token: `estás`

Etiquetas de token: `<MUTATE param="PERSON-3"/> <MUTATE param="MOOD-SUB"/> <MUTATE param="TIME-PRET"/>`

Por conveniencia, las representaciones numéricos de cada parte de las etiquetas son:
* MUTATE = 5
* PERSON-3 = 24
* MOOD-SUB = 26
* TIME-PRET = 31

Con max_labels = 5, la representación de vector de estas etiqueta son:

`[5, 24, 5, 26, 5, 31, 0, 0, 0, 0]`

y la resulta de estas transformaciones: `estuviera`

Tenga en cuenta que este proyecto es un trabajo en proceso, así que me alegraría mucho recibir comentarios para ajustar las definiciones del proyecto.

## Generar el Diccionario Morfológico

El diccionario morfológico se genera por JSON pre-preocesado de Wikitionary español, la base de datos de Jehle, y un archivo de entradas manuales. Se genera así:

```
python3 generate_morpho_dict.py [path to Wikitionary JSONL file] [path to Jehle verb database] --manual_entries manual_entries.json
```

El archivo de JSON de Wikitionary y la base de datos de Jehle son demasiados grandes de alojar en este Github. Si se necesita generar el diccionario morfológico, se puede descargar esos archivos aquí:

* [Wikitionary JSON](https://kaikki.org/dictionary/rawdata.html) (file: Spanish es-extract.jsonl (915.9MB))
* [Base de datos de Jehle](https://github.com/ghidinelli/fred-jehle-spanish-verbs/blob/master/jehle_verb_database.csv)

## Limitaciones

La limitación principial es que el algoritmo de decodificación es demasiado lento. Recomiendo que se ejecuta con más de 4 núcleos (yo usé 94) para evitar un largo tiempo de ejecución. Resulta por el uso del model de spaCy más grande (es_dep_news_trf) como una parte mayor del algoritmo. Tambíen hay alguna redundancia en que se necesita reusar el modelo de spaCy en una parte de la frase si cambia una palabra incluso si no es necesario. Por lo tanto, el primer cambio será una revisión del algoritmo de decodificación que 1) encadena mutaciones secuenciales, 2) no usa NLP si nunca se usará (i.e. no hay una mutación), y 3) ajusta el código para poder usar un modelo más pequeño (basado en reglas?) de morfología y etiquetación de POS (es_dep_news_sm?) si no afecta el rendimiento. En el futuro, es posible que reescribe este código para usar C++ o Rust por el algoritmo.
