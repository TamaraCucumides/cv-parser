import math
import nltk
import os
import itertools
import json
from gensim.models.keyedvectors import KeyedVectors
import pprint
from constantes import cargar_dict
from utils import similitud, palabras_cercanas, cosine_sim, sent2vec, lematizar, preprocesar_texto, eliminar_palabras_repetidas, stemizar
import numpy as np
import es_core_news_sm
import re



def load_embeddings():
    if os.path.isfile(os.getcwd() + '/embeddings/embeddings_pre_load'):
        model = KeyedVectors.load(os.getcwd() + '/embeddings/embeddings_pre_load', mmap='r')
    else:
        wordvectors_file_vec = os.getcwd() + '/embeddings/fasttext-sbwc.3.6.e20.vec'
        cantidad = 500000
        model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)
        model.init_sims(replace=True)
        model.save(os.getcwd() + '/embeddings/embeddings_pre_load')
    return model

def load_cv_json(path):
    # Se cargan todos los paths a los CV seccionados.
    path_to_json = os.getcwd() + path
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    cvs_seccionados = []
    for _, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            cvs_seccionados.append(json.load(json_file))
    return cvs_seccionados

def load_descripcion_cargo(path):
    # Se carga la descripción de cargo.
    file = os.getcwd() + path
    with open(file) as f:
        descripcion_cargo = " ".join([x.strip() for x in f]) 
    return descripcion_cargo

def load_stopwords_cargo(path):
    #Se cargan las stop_words especificas para descriptores 
    newStopWords = cargar_dict(os.getcwd() + path)
    stopwords = nltk.corpus.stopwords.words('spanish')
    stopwords.extend(newStopWords)
    return stopwords

def procesar_descripcion_cargo(descripcion_cargo, stopwords):
    nlp = es_core_news_sm.load()
    #pattern = r'[0-9]'
    # Se eliminan STOPWORDS -Puntuacion -numeros
    print("Descripcion original:")
    print(descripcion_cargo + '\n')
    
    #print(descripcion_cargo_lema + '\n')
    descripcion_cargo = preprocesar_texto(descripcion_cargo, stopwords, keepNumeros=False, keepPuntuacion=False) #eliminar stopword y a minusculas
    print("Descripcion procesada:")
    print(descripcion_cargo + '\n')
    
    descripcion_cargo = lematizar(descripcion_cargo, nlp) #lematizar
    print("Descripcion procesada y lematizada")
    print(descripcion_cargo+ '\n' )
    return descripcion_cargo

def expandir_descripcion(descripcion_cargo, model, num_palabras_similares = 0):
    word_value = {}
    for word in descripcion_cargo.split():
        palabras_similares, similarity = palabras_cercanas(word, num_palabras_similares, model)
        for i in range(len(palabras_similares)):
            word_value[palabras_similares[i]] = word_value.get(palabras_similares[i], 0)+similarity[i]
    print("Descripción expandida")
    print(word_value)
    return word_value

def get_idf(word_value, cvs_seccionados, stopwords, model):
    count = {}
    idf = {}
    no_of_cv = len(cvs_seccionados)
    for word in word_value.keys():
        count[word] = 0
        for i in range(no_of_cv):
            #Se eliminan STOPWORDS -Puntuacion
            skill_pro = ' '.join([str(x) for x in cvs_seccionados[i]['SKILLS'] + cvs_seccionados[i]['LICENCIA_CERTIFICACION']]) 
            expe_pro = preprocesar_texto(cvs_seccionados[i]['EXPERIENCIA'], stopwords, keepNumeros= False, keepPuntuacion= False)
            
            # En el caso que word se encuentre en skills o experiencia o eduacion
            # Se suma al contador
            if similitud(word, skill_pro.split(), model) or similitud(word, expe_pro.split(), model):
                count[word] += 1
                #print(word)

        # Se calcula idf con suavizado para evitar 0
        if count[word] != 0:
            idf[word] = math.log((no_of_cv+1)/(count[word]))
        else:
            idf[word] = -100
    return idf

def get_TF_IDF( stopwords, cvs_seccionados, word_value):
    no_of_cv = len(cvs_seccionados)
    score = {}
    for i in range(no_of_cv):
        score[i] = 0
        #Se eliminan STOPWORDS -Puntuacion
        skill_pro = ' '.join([str(x) for x in cvs_seccionados[i]['SKILLS'] + cvs_seccionados[i]['LICENCIA_CERTIFICACION']]) 
        expe_pro = preprocesar_texto(cvs_seccionados[i]['EXPERIENCIA'], stopwords, keepNumeros= False, keepPuntuacion= False)
        #edu_pro = preprocesar_texto(cvs_seccionados[i]['educación'], stopwords)
        n_words_skill =  len(skill_pro.split())
        n_words_expe_pro =  len(expe_pro.split())
        total_words = n_words_expe_pro
        for word in word_value.keys():
            # Se calcula tf como el número de veces que aparece una palabra en el CV. Donde
            # el criterio de aparecer se relaciona con una simulitud superior al umbral
            n_skills = similitud(word, skill_pro.split(), model)
            n_exp = similitud(word, expe_pro.split(), model)
            #n_edu = similitud(word, edu_pro.split(), model)

            #tf = 1 + n_skills +  n_edu + n_exp 
            if total_words == 0:
                score[i] += 0
            else:
                #tf = (n_skills*2.5 +  n_exp)/ total_words
                tf = (n_skills*5 +  n_exp/total_words)
                score[i] += word_value[word]*tf*idf[word]
    return score

def ordenar_ranking(score, cvs_seccionados):
    sorted_list = []
    no_of_cv = len(cvs_seccionados)
    for i in range(no_of_cv):
        sorted_list.append((np.around(score[i], decimals=4), cvs_seccionados[i]['NOMBRE_ARCHIVO']))

    # Se ordenan los puntajes de mayor a menor para mostrar los mejores
    sorted_list.sort(reverse = True)

    return sorted_list





#Load model embeddings 
model = load_embeddings()

#load specific stopwords
stopwords = load_stopwords_cargo('/diccionarios/stop_words_descripcion_cargo')

#load cv en formato .json
cvs_seccionados = load_cv_json('/Outputs/output_parser')

# load descriptor cargo y procesar
descripcion_cargo = load_descripcion_cargo('/diccionarios/descripcion_cargo')
descriptor_procesado = procesar_descripcion_cargo(descripcion_cargo, stopwords)

# expandir descripcion, con mejores embeddings puede ser util, por ahora no hace nada.
word_value = expandir_descripcion(descriptor_procesado, model, num_palabras_similares= 0)

#calcular TF-IDF
idf = get_idf(word_value, cvs_seccionados, stopwords, model)
score = get_TF_IDF( stopwords, cvs_seccionados, word_value)

# Se crea una lista con los puntajes y el respectivo nombre del CV
sorted_list = ordenar_ranking(score, cvs_seccionados)
pprint.pprint(sorted_list)

