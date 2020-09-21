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

import re



print("Cargando embeddings")
wordvectors_file_vec = os.getcwd() + '/embeddings/fasttext-sbwc.3.6.e20.vec'
cantidad = 200000
model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)
print("Embeddings cargadas" + '\n')



# Se cargan todos los paths a los CV seccionados.
#path_to_json = os.getcwd() + '/Outputs/output_seccionado'
path_to_json = os.getcwd() + '/Outputs/output_parser'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
cvs_seccionados = []
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        cvs_seccionados.append(json.load(json_file))



# Se carga la descripción de cargo.
file = os.getcwd() +'/diccionarios/descripcion_cargo'
with open(file) as f:
  descripcion_cargo = " ".join([x.strip() for x in f]) 

#Se cargan las stop_words especificas para descriptores 
newStopWords = cargar_dict(os.getcwd() + '/diccionarios/stop_words_descripcion_cargo')
stopwords = nltk.corpus.stopwords.words('spanish')
stopwords.extend(newStopWords)

pattern = r'[0-9]'
# Se eliminan STOPWORDS -Puntuacion -numeros
print(descripcion_cargo + '\n')
descripcion_cargo = re.sub(r'[^\w\s]',' ',descripcion_cargo) #eliminar puntuacion
descripcion_cargo = re.sub(pattern, ' ', descripcion_cargo) #eliminar numeros
descripcion_cargo_lema = lematizar(descripcion_cargo) #lematizar
print(descripcion_cargo_lema + '\n')
descripcion_cargo = preprocesar_texto(descripcion_cargo_lema, stopwords, numeros=False) #eliminar stopword y a minusculas
print(descripcion_cargo + '\n')
#descripcion_cargo = eliminar_palabras_repetidas(descripcion_cargo)
#print(descripcion_cargo + '\n')

des_cargo = []
des_stem = []
#word_stem_list = stemizar(descripcion_cargo.split())
for word in descripcion_cargo.split(" "):
    #print(word)
    word_stem = stemizar(word)
    
    if word_stem not in des_stem:
        des_cargo.append(word)
        des_stem.append(word_stem)
descripcion_cargo = " ".join(des_cargo)

#print(des_cargo)

print(descripcion_cargo)




#########################################################################################################
# Expandir descripcion usando palabra similares.
# word_value contendrá todas las palabras de la descripcion de cargo + sus similares
# de la siguiente forma: word_value{'palabra'} = similitud respecto a la original
# Por ejemplo si en la descripcion una de las palabras es "android" y guardamos 2 palabras parecidas:
# se guardaran 3 palabras, la orginal + 2.
# word_value{'android'} = 1, similitud consigo mismo
# word_value{'smartphone'} = 0.835 similitud con android
# word_valeu{'iphone'} = 0.82 similitud con android
##########################################################################################################


word_value = {}
num_palabras_similares = 0
for word in descripcion_cargo.split():
    palabras_similares, similarity = palabras_cercanas(word, num_palabras_similares, model)
    for i in range(len(palabras_similares)):
        word_value[palabras_similares[i]] = word_value.get(palabras_similares[i], 0)+similarity[i]

no_of_cv = len(cvs_seccionados)

print('______________________\n')
print(word_value)
# Se procede a calcular IDF

count = {}
idf = {}
for word in word_value.keys():
    count[word] = 0
    for i in range(no_of_cv):
        #Se eliminan STOPWORDS -Puntuacion
        #skill_pro = preprocesar_texto(cvs_seccionados[i]['Skills'], stopwords) 
        #skill_pro = cvs_seccionados[i]['Skills']
        skill_pro = ' '.join([str(x) for x in cvs_seccionados[i]['Skills'] + cvs_seccionados[i]['Licencias-Certificaciones']]) 
        expe_pro = preprocesar_texto(cvs_seccionados[i]['Experiencia'], stopwords, numeros= False)
        #print(skill_pro )
        #edu_pro = preprocesar_texto(cvs_seccionados[i]['educación'], stopwords)
        
        # En el caso que word se encuentre en skills o experiencia o eduacion
        # Se suma al contador
        #if similitud(word, skill_pro.split(), model) or similitud(word, expe_pro.split(), model) or similitud(word, edu_pro.split(), model):
        if similitud(word, skill_pro.split(), model) or similitud(word, expe_pro.split(), model):
            count[word] += 1
            #print(word)

    # Se calcula idf con suavizado para evitar 0
    if count[word] != 0:
        idf[word] = math.log((no_of_cv+1)/(count[word]))
    else:
        idf[word] = -100


#print('---------------\n \n \n')
# Calculo TF, y luego TF-IDF
score = {}
for i in range(no_of_cv):
    score[i] = 0
    #Se eliminan STOPWORDS -Puntuacion
    #skill_pro = preprocesar_texto(cvs_seccionados[i]['Skills'], stopwords) 
    skill_pro = ' '.join([str(x) for x in cvs_seccionados[i]['Skills'] + cvs_seccionados[i]['Licencias-Certificaciones']]) 
    expe_pro = preprocesar_texto(cvs_seccionados[i]['Experiencia'], stopwords, numeros= False)
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


# Se crea una lista con los puntajes y el respectivo nombre del CV
sorted_list = []
for i in range(no_of_cv):
    sorted_list.append((np.around(score[i], decimals=4), cvs_seccionados[i]['Nombre archivo']))

# Se ordenan los puntajes de mayor a menor para mostrar los mejores
sorted_list.sort(reverse = True)

pprint.pprint(sorted_list)

