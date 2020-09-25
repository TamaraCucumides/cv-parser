import re
import nltk
import pandas as pd
import os
import es_core_news_md
import itertools
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import json
import pprint
import string
pp = pprint.PrettyPrinter(indent=4)
import multiprocessing as mp
import time
import unidecode as u
import unicodedata

#from utils import preprocesar_texto
stopwords = nltk.corpus.stopwords.words('spanish')
# Utilidad para borrar simbolos
re_c = re.compile(r'\w+')

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

def keep_ene(string):
    #s1 = string.replace("ñ", "#").replace("Ñ", "%")
    #return unicodedata.normalize("NFKD", s1).encode("ascii","ignore").decode("ascii").replace("#", "ñ").replace("%", "Ñ")
    return string.strip()

def load_secciones(path):
    #Se carga la tabla de secciones.
    seccion_csv = os.getcwd() + path
    secciones = pd.read_csv(seccion_csv, header = 0)

    
    # Se carga el dccionario de secciones, se considera todas las celdas de seccion_csv que no sean nan.
    secciones_dict = {
        'EXTRAS' : [keep_ene(str(x.lower())) for x in secciones['EXTRAS'].values if str(x)!= 'nan'],
        'EXPERIENCIA' : [keep_ene(str(x.lower())) for x in secciones['EXPERIENCIA'].values if str(x)!= 'nan'],
        'EDUCACION' : [keep_ene(str(x.lower())) for x in secciones['FORMACION_ACADEMICA'].values if str(x)!= 'nan'],
        'SKILLS' : [keep_ene(str(x.lower())) for x in secciones['SKILLS'].values if str(x)!= 'nan']                   
            
    }
    similar_to = secciones_dict
    lista_secciones = secciones_dict.keys()
    # Se carga el modelo español de spacy
    nlp = es_core_news_md.load()
    # Usando secciones_dict que tiene las secciones a buscar y palabras que describen esas secciones
    # se llevan aquellas palabras a su lema
    for section in lista_secciones:
        new_list = []    
        for word in similar_to[section]:
            docx = nlp(word)
            new_list.append(docx[0].lemma_)
        

    similar_to[section] = list(set(new_list)) # se retorna lista de elementos unicos

    return lista_secciones, similar_to

def modificar(word, nlp):
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
    
def preprocesar_texto(corpus,stopwords = None , enminiscula= True, puntuacion = False, keepTildes = False):
    '''
    Entrada: texto, stopwords, enminiscula (opcional), puntuacion
    Salida:  texto
    Funcion que se encarga de limpiar las stopwords de un texto
    el parámetro opciona enminuscula si es verdadero,
    transforma todo el texto a miniscula y elimina stopwords que esten en minuscula.
    Cuando se usa false, el texto retornado mantendra capitalizacion original y 
    además se eliminan stop words especificas tales como: Pontificia, Universidad, Vitae, VITAE
    Notar que stop_words.txt tiene stopwords en minisculas y capitalizada.
    Esta propiedad de mantener la capitalización es útil en la detección de nombres.
    '''
    if stopwords is None:
        stopwords = nltk.corpus.stopwords.words('spanish')
    if enminiscula: # si se quiere normalizar a minuscula
        corpus = corpus.lower()
    if not puntuacion: # Si no se quiere conservar la puntuación
        stopset = stopwords+ list(string.punctuation)
    else:
        stopset = stopwords

    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    if not keepTildes:
        s1 = corpus.replace("ñ", "#").replace("Ñ", "%")
        corpus = unicodedata.normalize("NFKD", s1).encode("ascii","ignore").decode("ascii").replace("#", "ñ").replace("%", "Ñ")

    pattern = r'[0-9]'
    # Match all digits in the string and replace them by empty string
    corpus = re.sub(pattern, '', corpus)

    return corpus
 
def esta_vacia(line):
    '''
    Retorna un booleano correspondiendo a 
    si una linea esta vacia en términos de letras-números
    '''
    for c in line:
        if (c.isalpha()):
            return False
    return True

def seccionar_cv(path, model, lista_secciones, similar_to, nlp,threshold = 0.5):
    # Se crea un diccionario vacio para rellenarlo
    secciones_data = {
        'NOMBRE_ARCHIVO':'',
        'EXTRAS' : '',
        'EXPERIENCIA' : '',
        'EDUCACION' : '',
        'SKILLS':''}
    # Se carga un archivo .txt, que contiene el CV que venia del PDF
    close = True
    file = path
    try:
        cv_txt = open(file, "r")
    except:
        cv_txt = path
        close = False
    seccion_previa  = 'EXTRAS'

    for line in cv_txt:
        # si la linea esta vacia, entonces saltar
        if (len(line.strip()) == 0 or esta_vacia(line)):
            continue

        # procesar la linea
        palabras_en_linea = preprocesar_texto(line,puntuacion = False, keepTildes = True).split(" ")
        lista_palabras_utiles  = []
        
        # recorrer todas las palabras de linea actual
        for i in range(len(palabras_en_linea)):
            palabra_limpia = modificar(palabras_en_linea[i], nlp)

            if (palabra_limpia): 
                lista_palabras_utiles.append(palabra_limpia)
       
        linea_actual = ' '.join(lista_palabras_utiles) # crear un string a partir de una lista
        
        doc = nlp(linea_actual)
        valor_seccion = {}

        # Inicializar el valor de la seccion a 0
        for section in lista_secciones:
            valor_seccion[section] = 0.0
        valor_seccion[None] = 0.0

        # Actualizar los valores de las secciones   
        for token in doc:
            for section in lista_secciones:
                for word in similar_to[section]:
                    #print(word)
                    try:
                        if token.text == word:
                            sim = 1
                        else:
                            sim = float(model.similarity(token.text, word))
                        valor_seccion[section] = max(valor_seccion[section], sim)
                    except: 
                        pass # si es que token.text no esta en el vocabulario
                    
        # ver la siguiente sección probable de acuerdo al umbral establecido
        siguiente_seccion_prob = None
        for section in lista_secciones:
            if (valor_seccion[siguiente_seccion_prob] < valor_seccion[section] and valor_seccion[section] > threshold):
                siguiente_seccion_prob = section

        # Actualizar la seccion previa
        if (seccion_previa != siguiente_seccion_prob and siguiente_seccion_prob is not None):
            seccion_previa = siguiente_seccion_prob


        # Concatenar las palabras para formar un linea, las palabras se usan en su lema
 
        secciones_data[seccion_previa] += line + ' '
    if close:
        cv_txt.close()
    return secciones_data

def generate_json(cv, model, lista_secciones, similar_to, nlp):
    start_time = time.time()
    name = cv.replace(direc + dir_txt, '')        
    secciones_data = seccionar_cv(cv, model,lista_secciones, similar_to, nlp)
    secciones_data['NOMBRE_ARCHIVO'] = name

    with open(direc + dir_output + name+'.json', 'w',encoding='utf-8') as json_file:
        json.dump(secciones_data, json_file,ensure_ascii=False, indent=4)
    print(name, end=',')
    print(": %s seconds" % (time.time() - start_time))


    


if __name__ == '__main__':
    #pool = mp.Pool(mp.cpu_count())
    direc = os.getcwd()
    dir_txt = '/Outputs/output_text/'
    dir_output = '/Outputs/output_seccionado/'
    
    # Se cargan todos los paths a los cv en formato .txt 
    resumes_seccionado = []
    for root, _, filenames in os.walk(direc + dir_txt):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes_seccionado.append(file)

    print("Seccionando CVs: "+str(len(resumes_seccionado)))


    nlp = es_core_news_md.load()
    # Se carga el modelo de embeddings en español
    model = load_embeddings()
    # Se secciona cada CV
    lista_secciones, similar_to = load_secciones('/diccionarios/seccionesCV_similitud.csv')
    seccionados = [generate_json(cv, model, lista_secciones, similar_to, nlp) for cv in resumes_seccionado]
    
     

    print('Finalizado')





  