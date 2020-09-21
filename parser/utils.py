import re
from constantes import grados_educativos_orden, educacion, educacionSiglas, idiomas, idiomas_nivel, palabras_claves, licencias, cargar_dict
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
from nltk.stem import WordNetLemmatizer
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
    text = text.replace('-','')
    regex = re.compile(r"\d{9,11}")
    text = text.replace('-','') # eliminar guiones, que son necesarios para extraer links pero no aquí
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

def extraer_skills(text, nlp_text):
    
    '''
    Input: String de texto.
    Output: Lista con las skill encontradas.
    Funcion que busca los skill declarados del postulante.
    Se buscan tanto skill de 1 token como de varios.
    Hace uso del diccionario skills.txt.
    '''
    # eliminar stopwords
    tokens = [token.text for token in nlp_text if not token.is_stop]
    skills = cargar_dict(os.getcwd() +'/diccionarios/skills')
    skillset = []

    # revisar para palabras
    for token in tokens:
        token_un = unidecode.unidecode(token) # eliminar tildes
        if token_un.lower() in skills:
            skillset.append(token)
    
    # Revisar skills de más de una palabra.
    skills = [skill for skill in skills if len(skill.split())> 1]
    for item in skills:
        item_un = unidecode.unidecode(item)
        text_un = unidecode.unidecode(text.lower().replace('\n', ' '))
        if item_un.lower() in text_un:
                skillset.append(item)
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def extraer_licencias(text, nlp_text):
    '''
    Funcion que busca las licencias que declara
    el postulante
    '''
    # eliminar stopwords
    tokens = [token.text for token in nlp_text if not token.is_stop]
    licencias_dic = licencias
    licencias_list = [unidecode.unidecode(licencia.lower()) for licencia in licencias_dic]
    licencias_set = []
    # revisar para palabras
    for token in tokens:
        token_un = unidecode.unidecode(token) # eliminar tildes
        if token_un.lower() in licencias_list:
            licencias_set.append(token)
    
    # revisar frases
    licencias_list = [licencia for licencia in licencias_list if len(licencia.split())> 1]
    for item in licencias_list:
        item_un = unidecode.unidecode(item)
        text_un = unidecode.unidecode(text.lower().replace('\n', ' '))
        #este if es mas costoso pero más efectivo, a veces el nltk se come los 'de'
        if item_un.lower() in text_un:
                licencias_set.append(item)
        
    return [i.upper() for i in set([i.lower() for i in licencias_set])]

def extraer_educacion(text, nlp_text):
    '''
    Funcion que recupera las universidad o intituciones mencionadas en el CV
    Hace uso de 2 diccionarios: universidades.txt y universidades_siglas.txt
    Input: texto plano
    Output: Lista de strings unicos
    '''
    educacion_list=[]
    #Dejar solo sustantivos
    filter_noun = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text)) if pos[0] == 'N' or word == 'de']
   
    for item in educacion:
        #for noun in noun_chunks:
        item_un = unidecode.unidecode(item)
        text_un = unidecode.unidecode(text.lower().replace('\n', ' '))
        #este if es mas costoso pero más efectivo, a veces el nltk se come los 'de'
        if item_un.lower() in text_un:
            educacion_list.append(item)
                   
    for item in educacionSiglas:
        if item.upper() in filter_noun:
            educacion_list.append(item)

    # se usar set para crear una lista de elementos únicos
    unique_values = set(educacion_list)

    return list(unique_values) 

def extraer_idiomas(text, nlp_text):
    '''
    Funcion que recupera los idiomas que declara el postulante.
    Usa dos diccionarios: idiomas.txt e idiomas_nivel.txt
    Toma los elementos de estos diccionarios y computa
    todas las combinaciones. Luego estas combinaciones 
    son buscada en el texto.
    '''
    nlp = es_core_news_sm.load()
    # generamos el objeto nlp eliminando cualquier tipo de puntuación en el texto
    nlp_text = nlp(re.sub(r'[^\w\s]','',text))
    
    # uso de itertools para generar el cruze de los diccionarios
    combinaciones = list(itertools.product(idiomas, idiomas_nivel))
    combinaciones_strings = []
    for i in range(1, len(combinaciones)):
        #se concatena cada combinacion: inglés + ' ' + a1 = 'inglés a1' 
        combinaciones_strings.append(combinaciones[i][0] +' '+ combinaciones[i][1])

    # agregamos los idiomas por si solos : [inglés, francés, etc]
    combinaciones_strings = combinaciones_strings + idiomas
    noun_chunks = list(nlp_text.noun_chunks)
    
    #Ahora a buscar cada una de las combinaciones en cada frase o chunk
    # todo esto ignorando tildes
    idiomas_cv = []
    for item in combinaciones_strings:
        for noun in noun_chunks:
            item_un = unidecode.unidecode(item)
            noun_un = unidecode.unidecode(noun.text)
            if len(item_un.split())>1:
                if item_un.lower() in noun_un.lower() :
                    idiomas_cv.append(item.capitalize())
            else: 
                if item_un.lower() in  noun_un.lower().split() :
                    idiomas_cv.append(item.capitalize())
    return list(set(idiomas_cv))
 
def extraer_grado(text):
    '''
    Funcion que devuelve el grado más alto encontrado, depende de la 
    lista grados_educativos_orden
    Input: Texto Plano
    Output: Lista de strings
    '''
    education = []
    frases = sent_tokenize(text) # frases   

    for grado_name in grados_educativos_orden.keys():
        for grado in grados_educativos_orden[grado_name]:
            for frase in frases:
                # Eliminacion de tildes
                grado_un = unidecode.unidecode(grado)
                frase_un = unidecode.unidecode(frase)
                if (len(grado_un.lower().split()) == 1): # Si tenemos grados de 1 palabra
                    if grado_un.lower() in frase_un.lower().split(): # para que no considere webmaster como un master xd
                        education.append(grado_name.capitalize())

                elif grado_un.lower() in frase_un.lower(): # grados de varias palabras: administracion de empresas
                    education.append(grado_name.capitalize())
 
    # Como se buscaron en orden, si se detecta más de uno
    # se retorna el grado más alto, que debe estar el último de la lista
    if len(education)>0:
        education = [education[-1]]
    else:
        education = []

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
            linea_np = re.sub(r'[^\w\s]','', linea)
            referencia_np = re.sub(r'[^\w\s]','', referencia)
            linea_un = "".join(unidecode.unidecode(linea_np).split())
            referencia_un = "".join(unidecode.unidecode(referencia_np).split())
            if referencia_un.lower() == linea_un.lower():
                linea_referencia = True

        for otro in otros:
            otro_np = re.sub(r'[^\w\s]','', otro)
            linea_np = re.sub(r'[^\w\s]','', linea)
            otro_un = unidecode.unidecode(otro_np)
            linea_un = unidecode.unidecode(linea_np)

            if linea_un.lower() == otro_un.lower() and linea_referencia:
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
            linea_np = re.sub(r'[^\w\s]','', line)
            resumen_np = re.sub(r'[^\w\s]','', resumen)
            linea_un = "".join(unidecode.unidecode(linea_np).split())
            resumen_un = "".join(unidecode.unidecode(resumen_np).split())
            if resumen_un.lower() == linea_un.lower():
                n_linea = n
                break
 
    for line in cv_text.splitlines()[n_linea:-1]:   
        chunks = re.split(' +', line)
        linea =''
        for word in chunks:
            linea += word.lower() + ' '

        linea = " ".join(linea.split())
        for otro in otros:
            otro_np = re.sub(r'[^\w\s]','', otro)
            linea_np = re.sub(r'[^\w\s]','', linea)
            otro_un = "".join(unidecode.unidecode(otro_np).split())
            linea_un = "".join(unidecode.unidecode(linea_np).split())

            if linea_un.lower() == otro_un.lower() :
                siguiente_seccion = True
                break

        if siguiente_seccion:
            break

        if  siguiente_seccion == False:
            parrafo += linea + ' '
        
    parrafo = re.sub(r'\s+',' ', parrafo)
    return parrafo
   
def extraer_experiencia(cv_text, model):
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
            linea_np = re.sub(r'[^\w\s]','', linea)
            experiencia_np = re.sub(r'[^\w\s]','', experiencia)
            linea_un = "".join(unidecode.unidecode(linea_np).split())
            experiencia_un = "".join(unidecode.unidecode(experiencia_np).split())
            if experiencia_un.lower() == linea_un.lower():
                linea_experiencia = True

        for otro in otros:
            otro_np = re.sub(r'[^\w\s]','', otro)
            linea_np = re.sub(r'[^\w\s]','', linea)
            otro_un = "".join(unidecode.unidecode(otro_np).split())
            linea_un = "".join(unidecode.unidecode(linea_np).split())

            if linea_un.lower() == otro_un.lower() and linea_experiencia:
                siguiente_seccion = True
                break

        if siguiente_seccion:
            break

        if linea_experiencia == True and siguiente_seccion == False:
            parrafo += linea + ' \n'
    if len(parrafo.splitlines())<3:
        parrafo = seccionar_cv(cv_text, model)['Experiencia']
        #print(parrafo)

    parrafo = re.sub(r'\s+',' ', parrafo)
    return parrafo.replace('\n', ' ')

def extraer_nombre(text, nlp_text):
    '''
    Funcion que busca por 3 pronombres seguidos. Se recibe el texto
    plano  palabras que complican el analisis
    Input: texto plano
    Output: texto plano
    '''
    text = text[0:math.floor(len(text)/16)]
    nlp = es_core_news_sm.load()
    newStopWords = cargar_dict(os.getcwd() + '/diccionarios/stop_words_nombres')
    stopwords = nltk.corpus.stopwords.words('spanish')
    stopwords.extend(newStopWords)
    # Se procesa el 25% superior del texto. Se asume que el nombre deberia estar arriba
    # De forma empirirca, con mayusculas y con puntuación funciona mejor
    nlp_text = nlp(preprocesar_texto(text[0:math.floor(len(text)/4)],stopwords ,enminiscula= False,  puntuacion= True))
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

def preprocesar_texto(corpus,stopwords , enminiscula= True, puntuacion = False, numeros = True):
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
    if not puntuacion: # Si no se quiere conservar la puntuación
        stopset = stopwords+ list(string.punctuation)
    else:
        stopset = stopwords

    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    #corpus = unidecode.unidecode(corpus)

    if not numeros:
        corpus = ''.join([i for i in corpus if not i.isdigit()])
        corpus = " ".join(corpus.split())

    return corpus

def lematizar(frase):
    '''
    Esta función recibe un string y le aplica lematización:
    [correr corrido correrá] ---> [correr, correr, correr]
    '''
    nlp = es_core_news_sm.load()
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