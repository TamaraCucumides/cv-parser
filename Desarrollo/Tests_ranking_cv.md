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
cantidad = 300000

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
frase_1 = 'corredor de bolsa'
frase_2 = 'ingeniero comercial'


vector_sentence_1 = sent2vec(frase_1)
vector_sentence_2 = sent2vec(frase_2)
similitud = cosine_sim(vector_sentence_1,vector_sentence_2)

print(similitud)
```

```python
print(model.most_similar(positive=['excel', 'sap'], negative=['datos']))
```

```python
model.similar_by_word('corredor', 20)
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

test = pre_process('educación básica')
test

```

```python
test = lematizar('ingeniero')
test
```

```python
lematizar('ingeniero') in lematizar('ingenieria')
```

```python
model.similarity('ingeniero', 'ingenieria')
```

```python
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('spanish')
   # stemmed_claves = [stemmer.stem(token) for token in palabras_claves]
#stop_words = set(stopwords.words('spanish')) 

[stemmer.stem(x) for x in test] 
#stemmed_clave
```

```python
stemmer.stem('ingeniería') in stemmer.stem('ingeniería')
```

```python
import os
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize

def cargar_dict(path):
    with open(path) as f:  
        array = [x.strip() for x in f]
        c = [x for x in array if x != ''] # '' aparece cuando hay lines vacias
    return c
```

```python
newStopWords = cargar_dict('/home/erwin/Genoma/cv-parser/parser/diccionarios/stop_words')
stopwords = nltk.corpus.stopwords.words('spanish')
stopwords.extend(newStopWords)
```

```python



def pre_process(corpus, stopWords, enminiscula = True):
    if enminiscula:
        corpus = corpus.lower()
    stopset = stopwords+ list(string.punctuation)

    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    #corpus = unidecode.unidecode(corpus)
    return corpus
```

```python
pre_process('UNIVERSIDAD Universidad universidad', stopwords, False)
```

```python
import es_core_news_sm
def lematizar(frase):
    '''
    Esta función recibe un string y le aplica lematización:
    ingeniero ---> ingenier
    ingeniera ----> ingenier
    '''
    nlp = es_core_news_sm.load()
    doc = nlp(frase)
    lemmas = [tok.lemma_.lower() for tok in doc]
    return lemmas
```

```python
des = '''
Nos encontramos en la búsqueda de Socios comerciales, para administrar por completo importantes puntos de ventas (corner) de retail de calzado.

Has escuchado hablar de las franquicias y consignaciones? pues esta empresa trabaja a través de la consignación, esto quiere decir que la empresa te hace entrega del punto de venta, corriendo con todos los gastos de infraestructura, arriendo y toda la mercadería y renovación de stock constante, apoyo de visual y marketing,etc.
Buscamos vendedores que se encarguen de la administración por completo del punto de venta (RRHH,seguimiento y gestión inventarios, bodegas, ventas, etc.). Ofrecemos atractivas comisiones

Los montos líquidos PROMEDIO mensual a los que puedes acceder, posterior a los gastos de administración van desde: $600.000 a $1.000.000

Requisitos:
-Experiencia en ventas (dependiente o independiente)
-Disponibilidad para horarios de mall
-Motivación y orientación al logro y a la venta
-Ambición y competitividad
-Manejo conceptos básicos de contabilidad y legislación laboral
-Excelente capacidad de administrar un equipo.
-Posee una sociedad limitada (si no cuenta con esto, no es impedimento, puesto que es tramite simple que se realiza de manera on line)


Buscamos personas que deseen emprender, tengan orientación a la venta y el liderazgo, para todos ellos esta es la oportunidad para iniciar un negocio propio con el respaldo de una marca líder en su rubro.
'''


f =lematizar(des)
lematizado = ''
for word in f:
    lematizado += word +' '
lematizado.replace('\n', '')
```

```python
import os
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.matcher import Matcher
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

def pre_process(corpus,  enminiscula= True):
    '''
    Entrada: texto, stopwords, enminiscula (opcional)
    Salida:  texto
    Funcion que se encarga de limpiar las stopwords de un texto
    el parámetro opciona enminuscula si es verdadero,
    transforma todo el texto a miniscula y elimina stopwords que esten en minuscula.
    Cuando se usa false, el texto retornado mantendra capitalizacion original y 
    además se eliminan stop words especificas tales como: Pontificia, Universidad, Vitae, VITAE
    Notar que stop_words.txt tiene stopwords en minisculas y capitalizada.
    Esta propiedad de mantener la capitalización es útil en la detección de nombres.
    '''
    newStopWords = cargar_dict('/home/erwin/Genoma/cv-parser/parser/diccionarios/stop_words')
    stopwords = nltk.corpus.stopwords.words('spanish')
    print(newStopWords)
    stop = stopwords.extend(newStopWords)
    

    if enminiscula:
        corpus = corpus.lower()
    stopset = stopwords+ list(string.punctuation)

    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    #corpus = unidecode.unidecode(corpus)
    return corpus
```

```python
def cargar_dict(path):
    '''
    Utilidad para cargar los diccionarios
    '''
    with open(path) as f:  
        array = [x.strip() for x in f]
        c = [x for x in array if x != ''] # '' aparece cuando hay lines vacias
    return c

file = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1586903416737-NewHTJ'
cv_txt = cargar_dict(file)
```

```python
def modificar(word):
    '''
     Retornar el lema de la palabra. 
   Fijandose que no sean un simbolo raro ni una stopword.
    '''
    try:
        symbols = '''~`!@#$%^&*)(_+-=}{][|\:;",./<>?'''
        mod_word = ''
        
        for char in word:
            if (char not in symbols):
                mod_word += char.lower()

        docx = nlp(mod_word)

        if (len(mod_word) == 0 or docx[0].is_stop):
            return None
        else:
            return docx[0].lemma_
    except:
        return None # to handle the odd case of characters like 'x02', etc.
    

    

def esta_vacia(line):
    '''
    Retorna un booleano correspondiendo a 
    si una linea esta vacia en términos de letras-números
    '''
    for c in line:
        if (c.isalpha()):
            return False
    return True
```

```python
t = " ".join(cv_txt)

```

```python
def secciones_limpio(dataframe):
    ar = [str(exp).lower() for exp in dataframe if str(exp)!='nan' and str(exp)!= ' ']
    return ar
    
```

```python


path_secciones_dic = '/home/erwin/Genoma/cv-parser/parser/CSVs/Secciones CVs_buscador.csv'

secciones_dic = pd.read_csv(path_secciones_dic)
secciones_dic.head()
```

```python
experiencia = secciones_limpio(secciones_dic.Experiencia)
perfil = secciones_limpio(secciones_dic.Perfil)
educacion = secciones_limpio(secciones_dic.Educacion)
cursos = secciones_limpio(secciones_dic.Cursos)
habilidades = secciones_limpio(secciones_dic.Habilidades) 
contacto = secciones_limpio(secciones_dic.Contacto)
referencias = secciones_limpio(secciones_dic.Referencias)
logros = secciones_limpio(secciones_dic.Logros)
hobbies = secciones_limpio(secciones_dic.Hobbies)

otros = perfil + educacion + cursos + habilidades + contacto + referencias + logros+ hobbies
#otros
```

```python
def extraer_experiencia(cv_text):
    linea_experiencia = False
    siguiente_seccion = False
    parrafo = ''
    for line in cv_txt:
        line_np = re.sub(r'[^\w\s]','', line)
        l = sum([i.strip(string.punctuation).isalpha() for i in line_np.split()])
        chunks = re.split(' +', line)
        linea =''
        for word in chunks:
            linea += word.lower() + ' '

        if ((len(line.strip()) == 0 or l > 4) and linea_experiencia == False):
            continue


        linea = " ".join(linea.split())
        for experiencia in experiencia_list:
            linea_np = re.sub(r'[^\w\s]','', linea)
            experiencia_np = re.sub(r'[^\w\s]','', experiencia)
            linea_un = "".join(unidecode.unidecode(linea_np).split())
            experiencia_un = "".join(unidecode.unidecode(experiencia_np).split())
            if experiencia_un.lower() == linea_un.lower():
                #print('He pillado la seccion')
                linea_experiencia = True
                #continue


        for otro in otros:
            otro_np = re.sub(r'[^\w\s]','', otro)
            linea_np = re.sub(r'[^\w\s]','', linea)
            otro_un = unidecode.unidecode(otro_np)
            linea_un = unidecode.unidecode(linea_np)

            if linea_un.lower() == otro_un.lower() and linea_experiencia:
                #print(linea_un.upper())
                siguiente_seccion = True
                break

        if siguiente_seccion:
            break

        if linea_experiencia == True and siguiente_seccion == False:
            parrafo += linea + '\n'

    return parrafo
```

```python
path = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1586786838778-CV_Postulaciones'
path = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1571410211245-CV_Mariana_Wong'
path = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1571275460067-CV_Gonzalo_Vásquez'
path = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1569272625085-CV_Arianne.R.'
path = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1568522376573-CV_Jorge_Berna_Espinoza'
path = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1569025536095-19.08.19_CV_Nicolas_Achondo'
path = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1568672819667-CV_CAMILO_BUSTAMANTE_SANTANDERR'
path = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1566924682475-CV_CatalinaZunigaBilbao'
path = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1566746473048-Patricio_Mendez-CV'
file = path
cv_txt = open(file, "r")

extraer_experiencia(cv_txt)
```

```python
file = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1566924682475-CV_CatalinaZunigaBilbao'
cv_txt = open(file, "r").read()
#print(cv_txt)
def extraer_perfil(cv_text):
    otros = educacion + cursos + habilidades + contacto + referencias + logros+ hobbies + experiencia
    n = -1
    siguiente_seccion = False
    parrafo = ''
    #print(cv_text)
    #text_1 = cv_text
    n_linea = 0
    for line in cv_text.splitlines():
        n += 1
        #print(line)
        for resumen in perfil:
            linea_np = re.sub(r'[^\w\s]','', line)
            resumen_np = re.sub(r'[^\w\s]','', resumen)
            linea_un = "".join(unidecode.unidecode(linea_np).split())
            resumen_un = "".join(unidecode.unidecode(resumen_np).split())
            if resumen_un.lower() == linea_un.lower():
                linea_resumen = True
                print('pille linea resumen')
                print(resumen_un)
                n_linea = n
                break
    print(n_linea)
    #print(cv_txt)   
    for line in cv_text.splitlines()[n_linea:-1]:   
        #print('entre aca')
        chunks = re.split(' +', line)
        linea =''
        for word in chunks:
            linea += word.lower() + ' '

        linea = " ".join(linea.split())



        for otro in otros:
            otro_np = re.sub(r'[^\w\s]','', otro)
            linea_np = re.sub(r'[^\w\s]','', linea)
            otro_un = unidecode.unidecode(otro_np)
            linea_un = unidecode.unidecode(linea_np)

            if linea_un.lower() == otro_un.lower():
                #print(linea_un.upper())
                siguiente_seccion = True
                break

        if siguiente_seccion:
            break

        if  siguiente_seccion == False:
            parrafo += linea + '\n'

    return parrafo
    

    
```

```python


experiencia = secciones_limpio(secciones_dic.Experiencia)
perfil = secciones_limpio(secciones_dic.Perfil)
educacion = secciones_limpio(secciones_dic.Educacion)
cursos = secciones_limpio(secciones_dic.Cursos)
habilidades = secciones_limpio(secciones_dic.Habilidades) 
contacto = secciones_limpio(secciones_dic.Contacto)
referencias = secciones_limpio(secciones_dic.Referencias)
logros = secciones_limpio(secciones_dic.Logros)
hobbies = secciones_limpio(secciones_dic.Hobbies)

otros = perfil + educacion + cursos + habilidades + contacto +  logros+ hobbies + experiencia
```

```python

file = '/home/erwin/Genoma/cv-parser/parser/Outputs/output_text/1566924682475-CV_CatalinaZunigaBilbao'
cv_txt = open(file, "r").read()

def extraer_referencias(cv_text):
    linea_referencia = False
    siguiente_seccion = False
    parrafo = ''
    for line in cv_txt.splitlines():
        line_np = re.sub(r'[^\w\s]','', line)
        l = sum([i.strip(string.punctuation).isalpha() for i in line_np.split()])
        chunks = re.split(' +', line)
        linea =''
        for word in chunks:
            linea += word.lower() + ' '

        if ((len(line.strip()) == 0 or l > 4) and linea_referencia == False):
            continue

        #print(linea + '\n')
        linea = " ".join(linea.split())

        for referencia in referencias:
            linea_np = re.sub(r'[^\w\s]','', linea)
            referencia_np = re.sub(r'[^\w\s]','', referencia)
            linea_un = "".join(unidecode.unidecode(linea_np).split())
            referencia_un = "".join(unidecode.unidecode(referencia_np).split())
            if referencia_un.lower() == linea_un.lower():
                print('He pillado la seccion')
                linea_referencia = True
                #print(linea.UPPER())
                #continue


        for otro in otros:
            otro_np = re.sub(r'[^\w\s]','', otro)
            linea_np = re.sub(r'[^\w\s]','', linea)
            otro_un = unidecode.unidecode(otro_np)
            linea_un = unidecode.unidecode(linea_np)

            if linea_un.lower() == otro_un.lower() and linea_referencia:
                print(linea_un)
                siguiente_seccion = True
                break

        if siguiente_seccion:
            break

        if linea_referencia == True and siguiente_seccion == False:
            parrafo += linea + '\n'
            
    if len(parrafo.splitlines())>1:
        parrafo = "\n ".join([str(x) for x in parrafo.splitlines()[1:-1]])
        

    return parrafo

```

```python
print(extraer_referencias(cv_txt))
```

```python
print(cv_txt)
```

```python

```
