import math
import nltk
import os
import itertools
import json
from gensim.models.keyedvectors import KeyedVectors
import pprint
from cts import cargar_dict
from utils import calculo_similitud, get_closest, cosine_sim, sent2vec, lematizar, pre_process



# Se agregan STOP_WORDS desde el diccionario stop_words.txt
newStopWords = cargar_dict(os.getcwd() + '/parser/diccionarios/stop_words')
stopwords = nltk.corpus.stopwords.words('spanish')
stopwords.extend(newStopWords)


# Se carga el modelo de embeddings en español
wordvectors_file_vec = os.getcwd() + '/parser/embeddings/fasttext-sbwc.3.6.e20.vec'
cantidad = 100000
model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)




# Se cargan todos los paths a los CV seccionados.
path_to_json = os.getcwd() + '/parser/Outputs/output_seccionado'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
cvs_seccionados = []
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        cvs_seccionados.append(json.load(json_file))



# Se carga la descricpción de cargo.
file = os.getcwd() +'/parser/Descripcion_cargo/descripcion_cargo'
with open(file) as f:
  descripcion_cargo = " ".join([x.strip() for x in f]) 

# Se eliminan STOPWORDS -Puntuacion
descripcion_cargo = pre_process(descripcion_cargo, stopwords)
print(descripcion_cargo)




# Expandir descripcion usando palabra similares.
# word_value contendrá todas las palabras de la descripcion de cargo + sus similares
# de la siguiente forma: word_value{'palabra'} = similitud respecto a la original
# Por ejemplo si en la descripcion una de las palabras es "android" y guardamos 2 palabras parecidas:
# se guardaran 3 palabras, la orginal + 2.
# word_value{'android'} = 1, similitud consigo mismo
# word_value{'smartphone'} = 0.835 similitud con android
# word_valeu{'iphone'} = 0.82 similitud con android

word_value = {}
num_palabras_similares = 2
for word in descripcion_cargo.split():
    palabras_similares, similarity = get_closest(word, num_palabras_similares, model)
    for i in range(len(palabras_similares)):
        word_value[palabras_similares[i]] = word_value.get(palabras_similares[i], 0)+similarity[i]


no_of_cv = len(cvs_seccionados)




# Se procede a calcular IDF

count = {}
idf = {}
for word in word_value.keys():
    count[word] = 0
    for i in range(no_of_cv):
        #Se eliminan STOPWORDS -Puntuacion
        skill_pro = pre_process(cvs_seccionados[i]['skills'], stopwords) 
        expe_pro = pre_process(cvs_seccionados[i]['experiencia'], stopwords)
        edu_pro = pre_process(cvs_seccionados[i]['educación'], stopwords)
        
        # En el caso que word se encuentre en skills o experiencia o eduacion
        # Se suma al contador
        if calculo_similitud(word, skill_pro.split(), model) or calculo_similitud(word, expe_pro.split(), model) or calculo_similitud(word, edu_pro.split(), model):
            count[word] += 1

    # Se calcula idf con suavizado para evitar 0
    idf[word] = math.log((no_of_cv+1)/(1+count[word]))



# Calculo TF, y luego TF-IDF
score = {}
for i in range(no_of_cv):
    score[i] = 0
    #Se eliminan STOPWORDS -Puntuacion
    skill_pro = pre_process(cvs_seccionados[i]['skills'], stopwords) 
    expe_pro = pre_process(cvs_seccionados[i]['experiencia'], stopwords)
    edu_pro = pre_process(cvs_seccionados[i]['educación'], stopwords)

    for word in word_value.keys():
        # Se calcula tf como el número de veces que aparece una palabra en el CV
        n_skills = calculo_similitud(word, skill_pro.split(), model)
        n_exp = calculo_similitud(word, expe_pro.split(), model)
        n_edu = calculo_similitud(word, edu_pro.split(), model)

        tf = 1 + n_skills +  n_edu + n_exp 
        score[i] += word_value[word]*tf*idf[word]




# Se crea una lista con los puntajes y el respectivo nombre del CV
sorted_list = []
for i in range(no_of_cv):
    sorted_list.append((score[i], cvs_seccionados[i]['nombre archivo']))

# Se ordenan los puntajes de mayor a menor para mostrar los mejores
sorted_list.sort(reverse = True)

pprint.pprint(sorted_list)

