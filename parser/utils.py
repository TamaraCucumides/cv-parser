import re
from constantes import grados_educativos_orden, educacion, educacionSiglas, idiomas, idiomas_nivel, palabras_claves, licencias, cargar_dict, skills_dic
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
import itertools
from nltk.stem import SnowballStemmer
import textacy
import regex
import unidecode
from gensim.models.keyedvectors import KeyedVectors
import yaml
import numpy as np
import string
from constantes import perfil, educacion_sec, cursos, habilidades, contacto, logros, hobbies, experiencias, referencias
from seccionar import seccionar_cv
import os
import json
import textract
import nltk
import re
from nltk.stem import WordNetLemmatizer
from seccionar import load_secciones
import time
from spacy.matcher import PhraseMatcher
stemmer = SnowballStemmer('spanish')

#####################################################
####  UTILIDADES  parser.py  ########################
#####################################################
def validarString(s):
    '''
    Utilidad que verifica que un string 
    tiene al menos una letra o número
    "hi " ---> True
    "-----------" -> False
    '''
    letter_flag = False
    number_flag = False
    for i in s:
        if i.isalpha():
            letter_flag = True
        if i.isdigit():
            number_flag = True
    return letter_flag or number_flag

def extraer_texto(path):
    '''
    Utilidad para determinar para extraer
    texto. Llama a distintas funciones
    dependiendo de la extensión del archivo
    '''
    file_path = path.lower()
    if file_path.endswith('.pdf'):
        text = extraer_texto_pdf(path)
    elif file_path.endswith('.doc') or file_path.endswith('.docx'):
        text = extraer_texto_docx(path)
    else:
        text = None
    return text

def extraer_texto_docx(path):
    '''
    Utilidad para extraer texto desde
    .docx o .doc. Además se procesa para
    eliminar simbolos innecesarios y lineas
    redundantes.
    '''
    text = textract.process(path).decode('utf-8')
    text_2 = '' #texto sin mayusculas y saltos innecesarias 
    for line in text.splitlines():
        if not line.strip() or not validarString(line): #si la linea esta vacia, saltar
            continue
        line_2=''
        for word in line.split():
            if word.isupper(): # Si la palabra esta completamente en mayuscula
                line_2 += word.capitalize()+' ' # HOLA---> Hola

            else:
                line_2 += word+ ' '
        text_2 += " ".join(line_2.split()) +'\n'
    
    simbolos = ' -,\n./@' #Simbolos que se permiten, sirven para correo, links, etc.
    text_clean = ' '
    for char in text_2:
        if (char.isalnum())| (char in simbolos): #Si el char es alphanumerico o es un simbolo permitido
            text_clean += char
    
    text = text_clean
    #return re.sub(r'\n\s*\n', '\n',text)
    return text

def extraer_texto_pdf(path):
    '''
    Input: ruta hacia los archivos
    Salida: Texto plano como string
    Primero se extrae el texto usando fitz (PyMUPDF)
    Luego, debido a que la salida no es perfecta, se limpia
    eliminando todos los simbolos innecesarios.
    Además para efectos de deteccion de nombres, 
    palabras completamente en mayusculas se capitalizan
    sólo al principio. PAULA ---> Paula.
    Esto hace más robusto la detección de entidades.
    '''
    with fitz.open(path) as doc:
        text = ""
        for page in doc: # text contiene todo el texto extraido desde el PDF
            text += page.getText()
        
        text_2 = '' #texto sin mayusculas y saltos innecesarias 
        for line in text.splitlines():
            if not line.strip() or not validarString(line): #si la linea esta vacia, saltar
                continue
            line_2=''
            for word in line.split():
                if word.isupper(): # Si la palabra esta completamente en mayuscula
                    line_2 += word.capitalize()+' ' # HOLA---> Hola

                else:
                    line_2 += word+ ' '
            text_2 += " ".join(line_2.split()) +'\n'
        
        simbolos = ' -,\n./@' #Simbolos que se permiten, sirven para correo, links, etc.
        text_clean = ' '
        for char in text_2:
            if (char.isalnum())| (char in simbolos): #Si el char es alphanumerico o es un simbolo permitido
                text_clean += char
       
        text = text_clean
    return text

def extraer_mail(text):
    '''
    Input: Recibe texto plano
    Output: String que representa un mail.
    Se busca el mail usando una expresión regular
    texto....@....texto.
    En el caso de pillar más de una, se retorna la primera. 
    '''
    mails = re.findall(r'\S+@\S+', text)
    if len(mails)>1:
        mails  =mails[0]
    return mails

def extraer_fono(text):
    '''
    Retorna numero de 8-11 digitos.
    Input: Texto plano.
    Output: Texto plano.
    Busca 9 a 11 digitos seguidos.
    EL texto en que se busca no contiene espacios ni guiones
    En el caso de detectar más de uno, se retorna el primero encontrado
    '''
    #text = text.replace('-','')
    regex = re.compile(r"\d{9,11}")
    #text = text.replace('-','') # eliminar guiones, que son necesarios para extraer links pero no aquí
    text = preprocesar_texto(text, keepSimbolos = False)
    numbers_list=[]
    number = ''
    text_lines= text.splitlines()
    for line in text_lines:
        texto_busqueda = "".join(line.split())  #eliminar todos los espacios
        numbers = re.findall(regex, texto_busqueda)
        numbers_list +=  numbers

    if len(numbers_list) == 1:
        number = numbers_list[0]
    elif len(numbers_list)>1:
        number = max(numbers_list, key = len) #retornar el string más largo
  
    if len(number) > 10:
        number = '+'+number
    return number

def extraer_skills(text, nlp):
    
    '''
    Input: String de texto.
    Output: Lista con las skill encontradas.
    Funcion que busca los skill declarados del postulante.
    Se buscan tanto skill de 1 token como de varios.
    Hace uso del diccionario skills.txt.
    '''
    skillset = []
    matcher = PhraseMatcher(nlp.vocab)
    terms = [unidecode.unidecode(i.lower()) for i in skills_dic]
    # Only run nlp.make_doc to speed things up
    patterns = [nlp.make_doc(text.lower()) for text in terms]
    matcher.add("TerminologyList", None, *patterns)
    text = preprocesar_texto(text, keepTildes= False)
    doc = nlp(text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        skillset.append(span.text)
    return [i.upper() for i in set([i.lower() for i in skillset])]

def extraer_licencias(text, nlp):
    '''
    Funcion que busca las licencias que declara
    el postulante
    '''
    licencias_set = []
    matcher = PhraseMatcher(nlp.vocab)
    terms = [unidecode.unidecode(i.lower()) for i in licencias]
    # Only run nlp.make_doc to speed things up
    patterns = [nlp.make_doc(text.lower()) for text in terms]
    matcher.add("TerminologyList", None, *patterns)
    text = preprocesar_texto(text, keepTildes= False)
    doc = nlp(text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        licencias_set.append(span.text)
    return [i.upper() for i in set([i.lower() for i in licencias_set])]

def extraer_educacion(text, nlp):
    start_time = time.time()
    matcher = PhraseMatcher(nlp.vocab)
    educacion_list = []
    terms = [unidecode.unidecode(i.lower()) for i in educacion]
    # Only run nlp.make_doc to speed things up
    patterns = [nlp.make_doc(text.lower()) for text in terms]
    matcher.add("TerminologyList", None, *patterns)
    text = preprocesar_texto(text, keepTildes= False)
    doc = nlp(text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        educacion_list.append(span.text.upper())
    #print(": %s secon 2 " % (time.time() - start_time))
    #print(educacion_list)
    #print('____________________\n')
    uniques = list(set(educacion_list)) 
    #print(uniques)
    return uniques


def extraer_idiomas(text, nlp):
    '''
    Funcion que recupera los idiomas que declara el postulante.
    Usa dos diccionarios: idiomas.txt e idiomas_nivel.txt
    Toma los elementos de estos diccionarios y computa
    todas las combinaciones. Luego estas combinaciones 
    son buscada en el texto.
    '''
    
    idiomas_cv = []
    matcher = PhraseMatcher(nlp.vocab)
    terms = [unidecode.unidecode(i.lower()) for i in idiomas]
    # Only run nlp.make_doc to speed things up
    patterns = [nlp.make_doc(text.lower()) for text in terms]
    matcher.add("TerminologyList", None, *patterns)
    text = preprocesar_texto(text, keepTildes= False)
    doc = nlp(text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        idiomas_cv.append(span.text)
    return [i.upper() for i in set([i.lower() for i in idiomas_cv])]
 
def extraer_grado(text, nlp):
    '''
    Funcion que devuelve el grado más alto encontrado, depende de la 
    lista grados_educativos_orden
    Input: Texto Plano
    Output: Lista de strings
    '''
    start_time = time.time()
    education = []

    
    text = preprocesar_texto(text, keepTildes= False)
    doc = nlp(text)
    
    for grado_name in grados_educativos_orden.keys():
        matcher = PhraseMatcher(nlp.vocab)
        terms = [unidecode.unidecode(i.lower()) for i in grados_educativos_orden[grado_name]]
        # Only run nlp.make_doc to speed things up
        patterns = [nlp.make_doc(text.lower()) for text in terms]
        matcher.add("TerminologyList", None, *patterns)
        matches = matcher(doc)
        if matches:
            education.append(grado_name.upper())
    #print(education)
    # Como se buscaron en orden, si se detecta más de uno
    # se retorna el grado más alto, que debe estar el último de la lista
    if len(education)>0:
        education = [education[-1]]
    else:
        education = []
    #print(": %s seconfddddddddddds" % (time.time() - start_time))
    return education

def extraer_referencias(cv_txt):
    '''
    Utilidad que extrae las referencias profesionales
    que declara el postulante.
    '''
    otros = perfil + educacion_sec + cursos + habilidades + contacto +  logros+ hobbies + experiencias
    linea_referencia = False
    siguiente_seccion = False
    parrafo = ''
    for line in cv_txt.splitlines():
        #line_np = re.sub(r'[^\w\s]','', line)
        #l = sum([i.strip(string.punctuation).isalpha() for i in line_np.split()])
        chunks = re.split(' +', line)
        linea =''
        for word in chunks:
            linea += word.lower() + ' '

        if ((len(line.strip()) == 0 ) and linea_referencia == False):
            continue
        linea = " ".join(linea.split())

        for referencia in referencias:
            linea_np = preprocesar_texto(linea, keepTildes= False, keepSimbolos= False)
            referencia_np = preprocesar_texto(referencia, keepTildes= False, keepSimbolos= False)
            linea_un = "".join(linea_np.split())
            referencia_un = "".join(referencia_np.split())
            if referencia_un == linea_un:
                linea_referencia = True

        for otro in otros:
            #otro_np = re.sub(r'[^\w\s]','', otro)
            #linea_np = re.sub(r'[^\w\s]','', linea)
            otro_un = preprocesar_texto(otro, keepTildes= False, keepSimbolos= False)
            linea_un = preprocesar_texto(linea, keepTildes= False, keepSimbolos= False)

            if linea_un == otro_un and linea_referencia:
                siguiente_seccion = True
                break

        if siguiente_seccion:
            break

        if linea_referencia == True and siguiente_seccion == False:
            parrafo += linea + ' '

    parrafo = re.sub(r'\s+',' ', parrafo)
    return parrafo

def extraer_perfil(cv_text):
    '''
    Utilidad que extrae el resumen que suelen 
    poner los postulantes. 
    '''
    otros = educacion_sec + cursos + habilidades + contacto + referencias + logros+ hobbies + experiencias
    n = -1
    siguiente_seccion = False
    parrafo = ''
    n_linea = 0

    for line in cv_text.splitlines():
        n += 1
        for resumen in perfil:
            linea_np = preprocesar_texto(line, keepTildes= False, keepSimbolos= False)
            resumen_np = preprocesar_texto(resumen, keepTildes= False, keepSimbolos= False)
            linea_un = "".join(linea_np.split())
            resumen_un = "".join(resumen_np.split())
            if resumen_un == linea_un:
                n_linea = n
                break
 
    for line in cv_text.splitlines()[n_linea:-1]:   
        chunks = re.split(' +', line)
        linea =''
        for word in chunks:
            linea += word.lower() + ' '

        linea = " ".join(linea.split())
        for otro in otros:
            otro_np = preprocesar_texto(otro, keepTildes= False, keepSimbolos= False)
            linea_np = preprocesar_texto(linea, keepTildes= False, keepSimbolos= False)
            otro_un = "".join(otro_np.split())
            linea_un = "".join(linea_np.split())

            if linea_un == otro_un :
                siguiente_seccion = True
                break

        if siguiente_seccion:
            break

        if  siguiente_seccion == False:
            parrafo += linea + ' '
        
    parrafo = re.sub(r'\s+',' ', parrafo)
    return parrafo
   
def extraer_experiencia(cv_text, model, nlp):
    '''
    Utilidad que extrae la experincia laboral del
    postulante.
    '''
    
    cv_text = cv_text.splitlines()
    otros =  educacion_sec + cursos + habilidades + contacto + referencias + logros+ hobbies
    linea_experiencia = False
    siguiente_seccion = False

    parrafo = ''
    for line in cv_text:
        #line_np = re.sub(r'[^\w\s]','', line)
        chunks = re.split(' +', line)
        linea =''
        for word in chunks:
            linea += word.lower() + ' '

        if ((len(line.strip()) == 0  ) and linea_experiencia == False):
            continue

        linea = " ".join(linea.split())
        for experiencia in experiencias:
            linea_np = preprocesar_texto(linea, keepTildes= False, keepSimbolos= False)
            experiencia_np = preprocesar_texto(experiencia, keepTildes= False, keepSimbolos= False)
            linea_un = "".join(linea_np.split())
            experiencia_un = "".join(experiencia_np.split())
            if experiencia_un == linea_un:
                linea_experiencia = True

        for otro in otros:
            otro_np = preprocesar_texto(otro, keepTildes= False, keepSimbolos= False)
            linea_np = preprocesar_texto(linea, keepTildes= False, keepSimbolos= False)
            otro_un = "".join(otro_np.split())
            linea_un = "".join(linea_np.split())

            if linea_un== otro_un and linea_experiencia:
                siguiente_seccion = True
                break

        if siguiente_seccion:
            break

        if linea_experiencia == True and siguiente_seccion == False:
            parrafo += linea + ' \n'
    if len(parrafo.splitlines())<3:
        #start_time = time.time()
        lista_secciones, similar_to = load_secciones('/diccionarios/seccionesCV_similitud.csv')
        parrafo = seccionar_cv(cv_text, model,lista_secciones, similar_to, nlp,threshold = 0.5)['EXPERIENCIA']
        #print(": %s experiencia similutd seconds" % (time.time() - start_time))

    parrafo = re.sub(r'\s+',' ', parrafo)
    return parrafo.replace('\n', ' ')

def extraer_nombre(text, nlp_text, nlp):
    '''
    Funcion que busca por 3 pronombres seguidos. Se recibe el texto
    plano  palabras que complican el analisis
    Input: texto plano
    Output: texto plano
    '''
    text = text[0:math.floor(len(text)/16)]
    #start_time = time.time()
    newStopWords = cargar_dict(os.getcwd() + '/diccionarios/stop_words_nombres')
    stopwords = nltk.corpus.stopwords.words('spanish')
    stopwords.extend(newStopWords)
    #print(name, end=',')
    #print(": %s seconfddddddddddds" % (time.time() - start_time))
    # Se procesa el 25% superior del texto. Se asume que el nombre deberia estar arriba
    # De forma empirirca, con mayusculas y con puntuación funciona mejor
    nlp_text = nlp(preprocesar_texto(text[0:math.floor(len(text)/4)],stopwords ,enminiscula= False,  keepPuntuacion= True))
    NAME_PATTERN      = [{'POS': 'PROPN'}, {'POS': 'PROPN'},{'POS': 'PROPN'}]
    
    matcher = Matcher(nlp.vocab)
    pattern = [NAME_PATTERN]
    matcher.add('NAME', None, *pattern)
    
    matches = matcher(nlp_text)

    if matches == []:
        NAME_PATTERN      = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
        pattern = [NAME_PATTERN]
        matcher.add('NAME', None, *pattern)
        matches = matcher(nlp_text)

  
    nombre = ''
    for _, start, end in matches:
        span = nlp_text[start:end]
        # Capitalizar cada uno de los nombre, solo por comodidad al guardar, juan perez ----> Juan Perez
        for pronon in span.text.split():
            nombre += pronon.capitalize() + ' '
        return nombre

def extraer_linkedin(text):
    '''
    Funcion que captura el perfil de Linkedin del postulante
    se busca con una expression regular https://wwww.linkedin..........
    '''
    regex = re.compile(r"\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\"]))")
    links = re.findall(regex, text) # resultado de la forma ('www.linkedin.com/in/florenciavillegasduhau', '', '', '', ''), tupla de 4 elementos
    link_linkedin = None
    if links:
        for link in links:
            if 'linkedin' in link[0]:
                link_linkedin = link[0]

    return link_linkedin

def busqueda_palabras_claves(text):
    '''
    Funcion que busca palabras claves en el CV. Debido
    a que el candidato puede redactar sus cualidades. El
    texto se le aplica un stemmer. De forma de capturar variaciones
    de las palabras, en el contexto de redaccion.
    Ejemplo:
    proactividad --> proactiv
    proactivo --> proactiv
    liderar --> lider
    liderazgo --> liderazg
    liderar --> lider
    liderar --> lider
    puntual --> puntual
    puntualidad --> puntual
    paciencia --> pacienci
    '''

    stemmer = SnowballStemmer('spanish')

    word_tokens = word_tokenize(text) 
    stopwords = nltk.corpus.stopwords.words('spanish')
    stop_words = stopwords
    filtered_text = [w for w in word_tokens if not w in stop_words] 

    encontradas = []
    for palabra_clave in palabras_claves:
        stemmed_clave = stemmer.stem(palabra_clave)
        for word in filtered_text:
            if stemmer.stem(word).lower() == stemmed_clave.lower():
                encontradas.append(palabra_clave.capitalize())
    encontradas = list(set(encontradas))
    return encontradas if len(encontradas)>1 else None

#####################################################
######  UTILIDADES ranking.py  ######################
#####################################################
#stopwords_def = nltk.corpus.stopwords.words('spanish') # stopwords_defecto
def preprocesar_texto(corpus,stopwords = None , enminiscula= True, keepPuntuacion = False, keepNumeros = True, keepTildes= True, keepSimbolos = True):
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

    if enminiscula: # si se quiere normalizar a minuscula
        corpus = corpus.lower()

    if not keepPuntuacion and stopwords is not None: # Si no se quiere conservar la puntuación
        stopset = stopwords+ list(string.punctuation)
    elif not keepPuntuacion and stopwords is None:
        stopset = list(string.punctuation)
    elif keepPuntuacion and stopwords is not None:
        stopset = stopwords
    else:
        stopset = None

    
    if stopset is not None:
        corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    
    if not keepTildes:
        corpus = unidecode.unidecode(corpus)

    if not keepNumeros:
        corpus = ''.join([i for i in corpus if not i.isdigit()])
        corpus = " ".join(corpus.split())
    if not keepSimbolos:
        corpus = re.sub(r'[^\w\s]',' ',corpus)

    return corpus

def lematizar(frase, nlp):
    '''
    Esta función recibe un string y le aplica lematización:
    [correr corrido correrá] ---> [correr, correr, correr]
    '''
    doc = nlp(frase)
    lemmas = [tok.lemma_.lower() for tok in doc]
    lematizado = ''
    for word in lemmas:
        lematizado += word +' '
        lematizado.replace('\n', '')

    return lematizado

def sent2vec(s, model):
    '''
    Se obtiene la representación como vector
    para una frase.
    Para cada palabra se obtiene su vector.
    Luego se suman todos los vectores para obtener uno sólo.
    Este nuevo vector se divide por su nomra
    '''
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
    '''
    Cacula la similutud coseno
    '''
    return  np.dot(vec1,vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))

def palabras_cercanas(word, n, model):
    '''
    Funcion que retorna las n palabras mas cercana a n
    la funcion recibe la palabra, el número deseado y 
    el model a usar
    '''
    word = word.lower()
    #casos bases
    words = [word]
    similar_vals = [1]
    try:
        similar_list = model.most_similar(positive=[word],topn=n)
        
        for tupl in similar_list:
            words.append(tupl[0])
            similar_vals.append(tupl[1])
    except:# Si la palabra no existe en el vocabulario, retorna el caso base
        pass
    
    return words, similar_vals

def similitud(word1, array_palabras, model, threshold = 0.6):
    '''
    Funcion que se encarga de calcular
    la similitud de word1 con respecto
    a cada una de las palabras de array_palabras.
    Cuando una palabra supera al threshold se considera
    que son parecidas. Esta funcion es útil para
    calcular Tf-idf, ya que se busca palabras parecidas
    y no las palabras exactas.
    
    '''
    n_veces = 0
    for word in array_palabras:
        try:
            sim = model.similarity(word, word1)
            if sim > threshold:
                n_veces += 1
            else: # No se considera similar
                continue
        except: #No estaba la palabra
            pass
        
    return n_veces

def eliminar_palabras_repetidas(string):
    lista_unicos = []
    [lista_unicos.append(x) for x in string.split() if x not in lista_unicos]
    return " ".join(lista_unicos)

def stemizar(frase):    
    #frase = " ".join(lematizar(frase))
    nltk_tokens = nltk.word_tokenize(frase)
    stem = [stemmer.stem(tk) for tk in nltk_tokens]
    return " ".join(stem)