---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python [conda env:cv_parser] *
    language: python
    name: conda-env-cv_parser-py
---

```python
import re
import math
import fitz
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.matcher import Matcher
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
import es_core_news_sm
from nltk.tokenize import sent_tokenize
import os
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import es_core_news_sm
import itertools
from nltk.stem import SnowballStemmer
import textacy
import regex
import json
import unidecode
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

import yaml
wordvectors_file_vec ='/home/erwin/Genoma/cv-parser/parser/embeddings/fasttext-sbwc.3.6.e20.vec'
```

```python
cantidad = 700000

model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)
```

```python
path_to_json = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_seccionado'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
#print(json_files)  # for me this prints ['foo.json']
jsons = []
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        jsons.append(json.load(json_file))
        

```

```python
jsons[0]['experiencia']
```

```python
sent = pre_process(jsons[0]['experiencia'])
sent
```

```python
sent_lem = lematizar(sent)
print(sent_lem)
```

```python
lematizar('ingles')
```

```python
def sent2vec(s):
    '''Generate Vectora for sentences.'''
    M = []
    for w in s.split():
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v/np.sqrt((v**2).sum())

def cosine_sim(vec1, vec2):
    '''Return Cosine Similarity.'''
    return  np.dot(vec1,vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))

def get_closest(word, n):
    '''Get n most similar words by words.'''
    #This function can easily be expanded to get similar words to phrases--
    #using sent2vec() method defined in WithWord2Vec notebook. 
    word = word.lower()
    words = [word]
    similar_vals = [1]
    try:
        similar_list = model.most_similar(positive=[word],topn=n)
        
        for tupl in similar_list:
            words.append(tupl[0])
            similar_vals.append(tupl[1])
    except:
        #If word not in vocabulary return same word and 1 similarity-- 
        #see initialisation of words, similarities.
        pass
    
    return words, similar_vals
```

```python tags=[]
frase_1 = 'software developer'
frase_2 = 'web developer'


vector_sentence_1 = sent2vec(frase_1)
vector_sentence_2 = sent2vec(frase_2)
similitud = cosine_sim(vector_sentence_1,vector_sentence_2)

print(similitud)
```

```python
get_closest(word= 'ingle', n = 3)
```

```python
model.similarity('tradición', 'tradicional')
```

```python
import es_core_news_md
nlp = es_core_news_md.load()
nlp_text = nlp('educado')
nlp_text[0].lemma_
```

```python
prc_description = '''ingeniería máster postgrado excel desarrollo gestión comercial experiencia manejo clientes
emprendimiento liderar equipos planificar organizar dirigir trabajo presión seguimiento KPI inglés '''
```

```python
# https://github.com/prateekguptaiiitk/Resume_Filtering/blob/develop/Scoring/CV_ranking.ipynb
word_value = {}
similar_words_needed = 1
for word in prc_description.split():
    similar_words, similarity = get_closest(word, similar_words_needed)
    for i in range(len(similar_words)):
        word_value[similar_words[i]] = word_value.get(similar_words[i], 0)+similarity[i]
        #print(similar_words[i], word_value[similar_words[i]])
        #print('------------------------------------------------')
```

```python
word_value.keys()
```

```python
### ahora veamos si resulta el ranking
#frecuencia de término – frecuencia inversa de documento 
#Tf-idf
#Para calcular este ranking es mejor tener las secciones skills y experiencia
#con el fin de calcular esta metrica usando la ocurrencia de las palabras
# Tenemos todos los CV's y una descripción del cargo, a este descripcion del cargo
# tiene N palabras, le buscamos 2 palabras parecidas, generando una descripcion
# de N*2

# Usando esta nuevo set de palabras de descripcion, recorremos todos los cvs contando 
# la ocurrencia de estas palabras en cada documento, y luego se genera un ranking usando Tf-idf
# La pregunta es: ¿Lo haré sobre el documento entero? o ¿Trataré de seccionar y ocupar ciertas secciones?

no_of_cv = len(jsons)
#print(no_of_cv)

count = {}
idf = {}
for word in word_value.keys():
    count[word] = 0
    for i in range(no_of_cv):
        try:
            if word in jsons[i]['skills'] or word in jsons[i]['experiencia'] or word in jsons[i]['educación']:
                #print('entre')
                count[word] += 1
        except:
            pass

    idf[word] = math.log((no_of_cv+1)/(1+count[word]))
    #print(idf)
```

```python
#calculemos usando similitud



def calculo_similitud(word1, array_palabras):
    n_veces = 0
    for word in array_palabras:
        try:
            sim = model.similarity(word, word1)
            #print(word)
            #print(sim)
            if sim > 0.5:
                n_veces += 1
            else:
                continue
        except: #No estaba la palabra
            pass
        
    return n_veces



word1 = 'feliz'
array = ['alegre', 'contento', 'taladro', 'juguete']




n = calculo_similitud(word1, array)
n
```

```python
jsons[0]['skills'].split()
```

```python
no_of_cv = len(jsons)

count = {}
idf = {}
for word in word_value.keys():
    count[word] = 0
    for i in range(no_of_cv):
        # eliminación de stopwords y quizas lematizacion
        skill_pro = pre_process(jsons[i]['skills']) 
        expe_pro = pre_process(jsons[i]['experiencia'])
        edu_pro = pre_process(jsons[i]['educación'])
        
        if calculo_similitud(word, skill_pro.split()) or calculo_similitud(word, expe_pro.split()) or calculo_similitud(word, edu_pro.split()):
            count[word] += 1


    idf[word] = math.log((no_of_cv+1)/(1+count[word]))
```

```python
idf
```

```python

score = {}
for i in range(no_of_cv):
    score[i] = 0
    skill_pro = pre_process(jsons[i]['skills']) 
    expe_pro = pre_process(jsons[i]['experiencia'])
    edu_pro = pre_process(jsons[i]['educación'])
    for word in word_value.keys():
        
        
        
        tf = 1 + calculo_similitud(word, skill_pro.split()) + calculo_similitud(word, expe_pro.split()) + calculo_similitud(word, edu_pro.split())

        score[i] += word_value[word]*tf*idf[word]

print(score)
```

```python
sorted_list = []
for i in range(no_of_cv):
    sorted_list.append((score[i], jsons[i]['nombre archivo']))
    
sorted_list.sort(reverse = True)

```

```python
sorted_list
```

```python
sorted_list
```

```python

```

```python

```

```python
#sorted_list
```

```python

```

```python

```

```python
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import string
import spacy
import es_core_news_sm


def pre_process(corpus):
    corpus = corpus.lower()

    stopset = stopwords.words('spanish') + list(string.punctuation)

    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    corpus = unidecode(corpus)
    return corpus

def lematizar(frase):
    nlp = es_core_news_sm.load()
    doc = nlp(frase)
    lemmas = [tok.lemma_.lower() for tok in doc]
    return lemmas


```

```python




```

```python
sent = pre_process(prc_description)
sent_lem = lematizar(sent)
```

```python
sent
```

```python
sent_lem
```

```python
sent_2 = pre_process('un ingeniero puntual proactivo con experiencia de manejo de equipos, especialista en redes')
sent_lem_2 = lematizar(sent_2)
sent_2
```

```python
sent_lem_2
```

```python

```

```python
model.similarity('especialista', 'experto')
```

```python
file = '/home/erwin/Genoma/cv-parser/parser/Descripcion_cargo/descripcion_cargo'
cv_txt = open(file, "r")
print(cv_txt.read())
```

```python
d = cv_txt.read()
```

```python
" ".join([x.strip() for x in cv_txt]) 
```

```python
with open(file) as f:
  descripcion = " ".join([x.strip() for x in f]) 
```

```python
descripcion
```

```python
cv_txt
```

```python

```
