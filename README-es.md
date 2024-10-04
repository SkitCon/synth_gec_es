# synth_gec_es
[![en](https://img.shields.io/badge/lang-en-red.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-yellow.svg)](README-es.md)

La corrección de errores gramaticales sintéticos para ESpañol es un sistema para generar datos GEC syntéticos para errors gramaticales comúnes en español para entrenar un modelo GEC.

Generalmente se usa para amplificar conjuntos de datos más pequeños de alta calidad igual que en *GECToR – Grammatical Error Correction: Tag, Not Rewrite* (2020)

**TRABAJO EN PROCESO**

## Índice de Materias
* [Scripts](#scripts)
  * [generate.py](#generate.py)
  * [parse.py](#parse.py)
  * [label.py](#label.py)
* [Definiciones](#definiciones)
* [Tipos de Mutación](#tipos-de-mutación)
* [Etiquetas Múltiples](#etiquetas-múltiples)

## Scripts

### generate.py

El script principal es generate.py. Se ejecuta como:

```
python generate.py INPUT_FILE [OUTPUT_FILE] [-n/--num-sentences] [number to generate for each origin] [-s/--seq2seq]  [-t/--token] [move-absolute, move-relative, replace]
```

Este script se usa para generar frases con errores de un archivo de corpus con frases sin errores.

* input file es una ruta de entrada a un archivo .txt con una frase por cada línea. Tenga en cuenta que este script espera datos sin etiqueta.
* output file es opcional, define la ruta de salida para guardar los datos sintéticos. Por omisión, la ruta de salida es [input file name]_synth.txt.
* --num-sentences es el número de frases con errores que se generará de cada frase original en el corpus de entrada. Por omisión es 1.
* --seq2seq significa que los datos sintéticos de salida van a incluir las frases con errores sin etiquetas para usarlas en un sistema GEC tradicional basado en NMT (e.g. con BART o T5)
* --token significa que los datos sintéticos de salido van a incluir las etiquetas de token para las frases con errores (lea más [abajo](#definiciones)) para usarlas en un sistema GEC de token (e.g. GECToR)
  * --token requiere por los menos un argumento asociado: move-absolute, move-relative, o replace (una frase con errores para cada argumento asociado)
    * move-absolute signfica que la etiqueta *MOVE* se incluirá and usará índices *absolutos* para calcular la posición final del token. Esta posición final se calcula **finalmente**.
    * move-relative significa que la etiqueta *MOVE* se incluirá and usará índices *relativos* para calcular la posición final del token. La posición final se calcula **finalmente** y de **izquierda a derecha**.
    * replace significa que la etiqueta *REPLACE* se incluirá en vez de *move*
   
### parse.py

Se ejecuta como:

```
python3 parse.py INPUT_FILE [OUTPUT_FILE] [-d/--dictionary-file] [dictionary file] [-v/--vocab-file] [vocab file] [-t/--token] [move-absolute, move-relative]
```

Este script se usa un archivo con frases con errores + etiquetas de token para convertirlas a frases corregidas.

* input file es una ruta a un archivo txt con una frase con errores en una línea, las etiquetas de token en la línea siguiente, y una línea en blanco entre esta frase y la siguiente
* output file es opcional, defines la ruta de salida para guardar las frases corregidas. Por omisión es [input file name]_parsed.txt.
* --dictionary-file es la ruta al archivo del diccionario que define las formas morfologías para una palabra
* --vocab-file es la ruta al archivo de vocabulario que contiene todas las palabras en el vocabulario de tu modelo
* --token define el tipo de índice para *MOVE* (lea más [abajo](#definiciones))
  * move-absolute signfica que la etiqueta *MOVE* se incluirá and usará índices *absolutos* para calcular la posición final del token. Esta posición final se calcula **finalmente**.
  * move-relative significa que la etiqueta *MOVE* se incluirá and usará índices *relativos* para calcular la posición final del token. La posición final se calcula **finalmente** y de **izquierda a derecha**

### label.py

Se ejecuta como:

```
python3 label.py INPUT_FILE [OUTPUT_FILE] [-d/--dictionary-file] [dictionary file] [-v/--vocab-file] [vocab file] [-t/--token] [move-absolute, move-relative, replace]
```

Este script se usa frases con errores + frases sin errores y crea etiquetas de token para ellas.

* input file es una ruta a un archivo txt con una frase con errores en una línea, la frase sin errores en la siguiente, y una línea en blanco entre esta frase y la siguiente
* output file es opcional, defines la ruta de salida para guardar las etiquetas de token. Por omisión es [input file name]_labeled.txt.
* --dictionary-file es la ruta al archivo del diccionario que define las formas morfologías para una palabra
* --vocab-file es la ruta al archivo de vocabulario que contiene todas las palabras en el vocabulario de tu modelo
* --token define el tipo de índice para *MOVE* (lea más [abajo](#definiciones))
  * move-absolute signfica que la etiqueta *MOVE* se incluirá and usará índices *absolutos* para calcular la posición final del token. Esta posición final se calcula **finalmente**.
  * move-relative significa que la etiqueta *MOVE* se incluirá and usará índices *relativos* para calcular la posición final del token. La posición final se calcula **finalmente** y de **izquierda a derecha**

## Definiciones

Las etiquetas de token principales son:
* `<KEEP/>`
  * No cambiar el token
* `<DELETE/>`
  * Borrar el token
* `<PRE-ADD token=i/>`
  * Añadir el token *i* (por índice de token en vocab.txt) justo antes de este token
* `<POST-ADD token=i/>`
  * Añadir el token *i* (por índice de token en vocab.txt) justo después de este token
* `<MUTATE type=x>`
  * Mutar la morfología de este token basado en el tipo *x*. Los tipos principales son CAPITALIZE, GENDER, NUMBER, PERSON, MOOD, y TIME. Más información [aquí](#tipos-de-mutación)
* `<REPLACE token=i/>`
  * Reemplacar este token con el token *i* (por índice de token en vocab.txt)
* `<MOVE pos=i/>`
  * Mover este token a la posición *i* (en el caso de índices absolutos) o mover este token *i* posiciones (en el caso de índices relativos).
    * En ambos casos, el índice se basa en la posición de los tokens después de todos los cambios. En el caso de índices relativos, se mueve de izquierda a derecha.

## Tipos de Mutación
* CAPITALIZE (ESCRIBIR CON MAYÚSCULAS)
  * TRUE - escribir la palabra con mayúsculas
  * FALSE - escribir la palabra con minúsculas
* GENDER (GÉNERO)
  * MASC - convertir la palabra al masculino
    * e.g. apuesta + `<MUTATE type=GENDER-MASC/>` = apuesto
  * FEM - convertir la palabra al feminino
    * e.g. harto + `<MUTATE type=GENDER-MASC/>` = harta
* NUMBER (NÚMERO)
  * SING - convertir la palabra al singular
  * PLU - convertir la palabra al plural
* PERSON (PERSONA)
  * 1 - convertir la palabra a la primera persona
    * e.g. son + `<MUTATE type=PERSON-1` = somos
  * 2 - convertir la palabra a la segunda persona
  * 3 - convertir la palabra a la tercera persona
* MOOD (MODO)
  * IND (INDICATIVO) - convertir la palabra al indicativo
  * POS-IMP (IMPERATIVO AFIRMATIVO) - convertir la palabra al imperativo afirmativo
  * NEG-IMP (IMPERATIVO NEGATIVO) - convertir la palabra al imperativo negativo
  * SUBJ (SUBJUNTIVO) - convertir la palabra al subjuntivo
  * PROG (PROGRESIVO) - convertir la palabra al progresivo (con la conjugación correcta de *estar* antes del verbo)
  * PERF (PERFECTO) - convertir la palabra al perfecto (con la conjugación correcta de *haber* antes del verbo)
  * PERF-SUBJ (PERFECTO SUBJUNCTIVO) - convertir la palabra al perfecto subjuntivo (con la conjugación correcta de *haber* antes del verbo)
  * GER (PARTICIPIO PRESENTE) - convertir la palabra al participio presente, distinto del progresivo que también añade la conjugación correcta *estar*
  * PAST-PART (PARTICIPIO PASADO) - convertir la palabra al participio pasado
  * INF (INFINITIVO) - convertir la palabra al infinitivo (i.e. sin modo)
* TIME (TIEMPO)
  * PRES (PRESENTE)
  * PRET (PRETÉTERITO)
  * IMPERF (IMPERFECTO)
  * COND (CONDICIONAL)
  * FUT (FUTURO)

## Etiquetas Múltiples
Es posible que hayas dado cuenta de que estas etiqueta no son mutuamente excluyentes. Algunos de las etiqueta no son compatibles o que en conjunto resultan redundantes, pero muchas se esperan operar en conjunto. Por los tanto, la etiquetas se separa con pestañas. Si no hay una pestañas entre dos etiquetas, estas etiqueta aplica al mismo token. Por ejemplo:

```
espero que corre tú bien.
<MUTATE type=CAPITALIZE-TRUE/>  <KEEP/>  <MOVE i=1/><MUTATE type=PERSON-2/><MUTATE type=MOOD-SUBJ/>  <KEEP/>  <KEEP/>  <KEEP/>
```

convierte la frasa al:

`Espero que tú corras bien.`

Por lo tanto, es clasificación multietiqueta para cada token. La salida esperada para cada token es un vector de integer con la longitud de etiquetas posibles (30). Cada dimensión es binaria **excepto** que las que representan el MOVE (value = índice), PRE-ADD/POST-ADD (value = índice del token), y REPLACE (value = índice del token).

Tenga en cuente que este proyecto es un trabajo en proceso, así que me alegraría mucho recibir comentarios para ajustar las definiciones del proyecto. Es mi primera iteración y ya no tengo evaluaciónes de rendimimento con este diseño del sistema.
