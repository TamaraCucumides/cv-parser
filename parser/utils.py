import re
from cts import grados_educativos_orden, educacion, educacionSiglas, idiomas, idiomas_nivel, palabras_claves, licencias
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
from cts import cargar_dict
from gensim.models.keyedvectors import KeyedVectors
import yaml
import numpy as np
import string

# Se agregan STOP_WORDS desde el diccionario stop_words.txt
newStopWords = cargar_dict(os.getcwd() + '/parser/diccionarios/stop_words')
stopwords = nltk.corpus.stopwords.words('spanish')
stopwords.extend(newStopWords)


#####################################################
####  UTILIDADES generate_text_files.py  ############
#####################################################

def extract_text(path):
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
            if not line.strip(): #si la linea esta vacia, saltar
                continue
            line_2=''
            for word in line.split():
                if word.isupper(): # Si la palabra esta completamente en mayuscula
                    line_2 += word.capitalize()+' '

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




#####################################################
####  UTILIDADES  parser.py  ########################
#####################################################


def retrieve_email(text):
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


def retrieve_phone_number(text):
    '''
    Retorna numero de 8-11 digitos
    Input: Texto plano
    Output: Texto plano
    Busca 8 a 9 digitos seguidos.
    EL texto en que se busca no contiene espacios ni guiones
    En el caso de detectar más de uno, se retorna el primero encontrado
    '''
    text = text.replace('-','')
    #regex = re.compile(r"\+?\d[\( -]?\d{3}[\) -]?\d{3}[ -]?\d{2}[ -]?\d{2}")
    regex = re.compile(r"\d{8,11}")
    text = text.replace('-','') # eliminar guiones, que son necesarios para extraer links pero no aquí
    texto_busqueda = "".join(text.split())  #eliminar todos los espacios
    numbers = re.findall(regex, texto_busqueda)
    if len(numbers)>1:
        numbers = numbers[0]

    return numbers


def retrieve_skills(nlp_text):
    '''
    Funcion que busca los skill declarados del postulante
    Se buscan tanto skill de 1 token como de varios.
    Hace uso del diccionario skills.txt.

    '''
    # eliminar stopwords
    tokens = [token.text for token in nlp_text if not token.is_stop]
 
    skills = cargar_dict(os.getcwd() +'/parser/diccionarios/skills')

    skillset = []
    # lista de frases
    noun_chunks = list(nlp_text.noun_chunks)


    # revisar para palabras
    for token in tokens:
        token_un = unidecode.unidecode(token) # eliminar tildes
        if token_un.lower() in skills:
            skillset.append(token)
    
    # revisar frases
    for chunk in noun_chunks:
        st = chunk.text
        chunk_lower = st.lower()  
        for skill in skills:
                skill_un = unidecode.unidecode(skill)
                chunk_un = unidecode.unidecode(chunk_lower)
                if skill_un in chunk_un: # si el skill se encuentra dentro de la frase
                    skillset.append(skill.capitalize())
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def retrieve_education_institution(text, nlp_text):
    '''
    Funcion que recupera las universidad o intituciones mencionadas en el CV
    Hace uso de 2 diccionarios: universidades.txt y universidades_siglas.txt
    Input: texto plano
    Output: Lista de strings unicos
    '''
   
    educacion_list=[]

    #Dejar solo sustantivos
    filter_noun = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text)) if pos[0] == 'N']
    #frases   
    noun_chunks = list(nlp_text.noun_chunks)
    
    for item in educacion:
        for noun in noun_chunks:
            item_un = unidecode.unidecode(item)
            noun_un = unidecode.unidecode(noun.text)
            if item_un.lower() in " ".join(noun_un.lower().split()):
                educacion_list.append(item)

                
    for item in educacionSiglas:
        if item in filter_noun:
            educacion_list.append(item)

    # se usar set para crear una lista de elementos únicos
    unique_values = set(educacion_list)

    return list(unique_values) 
 

def retrieve_languages(text, nlp_text):
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
            if item_un.lower() in noun_un.lower() :
                idiomas_cv.append(item.capitalize())
    return list(set(idiomas_cv))


    




def retrieve_higher_degree(text):
    '''
    Funcion que devuelve el grado más alto encontrado, depende de la 
    lista grados_educativos_orden
    Input: Texto Plano
    Output: Lista de strings
    '''
    education = []
    frases = sent_tokenize(text) # frases


    for grado in grados_educativos_orden:
         for frase in frases:
            # Eliminacion de tildes
            grado_un = unidecode.unidecode(grado)
            frase_un = unidecode.unidecode(frase)
            if (len(grado_un.lower().split()) == 1): # Si tenemos grados de 1 palabra
                if grado_un.lower() in frase_un.lower().split(): # para que no considere webmaster como un master xd
                    education.append(grado.capitalize())

            elif grado_un.lower() in frase_un.lower(): # grados de varias palabras: administracion de empresas
                education.append(grado.capitalize())


 
    # Como se buscaron en orden, si se detecta más de uno
    # se retorna el grado más alto, que debe estar el último de la lista
    if len(education)>0:
        education = [education[-1]]
    else:
        education = []

    return education


def retrieve_dates(text):
    pass


def retrieve_past_experience(text): # Funcion que no  usada
                                    # muy poco robusta

    
    '''
    Busca la palabra experiencia y devuelve la frase
    en la que esta incluida.
    :param resume_text: Plain resume text
    :return: list of experience
    '''
   
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('spanish'))

    # word tokenization
    word_tokens = nltk.word_tokenize(text)

    # remove stop words and lemmatize
    filtered_sentence = [
            w for w in word_tokens if w not
            in stop_words and wordnet_lemmatizer.lemmatize(w)
            not in stop_words
        ]
    sent = nltk.pos_tag(filtered_sentence)

    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)

    test = []

    for vp in list(
        cs.subtrees(filter=lambda x: x.label() == 'P')
    ):
        test.append(" ".join([
            i[0] for i in vp.leaves()
            if len(vp.leaves()) >= 2])
        )

    x = [x[x.lower().index('experiencia') + 12:]
        for i, x in enumerate(test)
        if x and 'experiencia' in x.lower()
    ]
    return x if len(x)>0 else None


def retrieve_experience_2(text): # Funcion no usada
    '''
    Funcion que busca palabras claves en las
    frases detectadas en el texto plano. Si es 
    que se encuentra "experiencia laboral", entonces
    se busca en que posicion comienza el match.
    De forma de devolver "experiencia laboral ........"
    '''
    text = ' '.join(text.split())
    nlp = textacy.load_spacy_lang('es_core_news_sm')
    texto_procesado = nlp(text)
    word_key = ['experiencia laboral', 'experiencia', 'experiencia profesional']
    experiencia = []
    for sent in texto_procesado.sents:
        for word in word_key:
            frase = sent.string.replace("\n","").lower().replace(" ", "")        # sin espacios, sin saltos y en minuscula
            word_search = word.replace(" ", "").lower()
            if word_search in frase:
                pos = sent.string.lower().find(word_search)                      # posicion match
                string_experiencia = sent.string[pos:]
                if len(string_experiencia)> 1:
                    text = string_experiencia
                    text = text.replace("•", "")
                    text = text.replace("▪", "")
                    text = text.replace("-", "")
                    experiencia.append(text.replace("\n"," "))
            
            
    return list(set(experiencia))
    


def retrieve_last_experience_year(text): #Funcion no usada
    pass



def retrieve_name(text, nlp_text):

    '''
    Funcion que busca por 3 pronombres seguidos. Se recibe el texto
    plano  palabras que complican el analisis
    Input: texto plano
    Output: texto plano
    '''
    text = text[0:math.floor(len(text)/16)]
    text = pre_process(text, stopwords, False)
    NAME_PATTERN      = [{'POS': 'PROPN'}, {'POS': 'PROPN'},{'POS': 'PROPN'}]
    nlp = es_core_news_sm.load()
    matcher = Matcher(nlp.vocab)
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


def parse_cv_sections(text): # No usada
    pass


def summarize_cv(text, nlp_text): # No usada
    '''
    Funcion que que rankea frases a partir de frecuencia
    de palabras, es un intento simple/ no muy efectivo de resumir.
    El problema de los cv es que el texto es reducido, no hablamos de un libro.

    Input: texto plano
    Output: texto plano
    '''
    corpus = [sent.text.lower() for sent in nlp_text.sents ]
    STOP_WORDS = set(stopwords.words("spanish"))
    cv = CountVectorizer(stop_words=list(STOP_WORDS))   
    cv_fit=cv.fit_transform(corpus)    
    word_list = cv.get_feature_names()    
    count_list = cv_fit.toarray().sum(axis=0)
    word_frequency = dict(zip(word_list,count_list))
    val=sorted(word_frequency.values())

    # gets relative frequencies of words
    higher_frequency = val[-1]

    for word in word_frequency.keys():  
        word_frequency[word] = (word_frequency[word]/higher_frequency)
        
    sentence_rank={}
    for sent in nlp_text.sents:
        for word in sent :       
            if word.text.lower() in word_frequency.keys():            
                if sent in sentence_rank.keys():
                    sentence_rank[sent]+=word_frequency[word.text.lower()]
                else:
                    sentence_rank[sent]=word_frequency[word.text.lower()]
    top_sentences=(sorted(sentence_rank.values())[::-1])
    top_sent=top_sentences[:1]


    summary=[]
    for sent,strength in sentence_rank.items():  
        if strength in top_sent:
            summary.append(sent)
        else:
            continue

    return summary[0].text.replace("\n","")

def extract_linkedin(text):
    '''
    Funcion que captura el perfil de Linkedin del postulante
    se busca con una expression regular https://wwww.linkedin..........
    '''
    regex = re.compile(r"(?:https?:)?\/\/(?:[\w]+\.)?linkedin\.com\/in\/(?P<permalink>[\w\-\_À-ÿ%]+)\/?")
    profile = re.findall(regex, text)
    if profile:
        return 'https://www.linkedin.com/in/' + profile[0]
    else:
        return None


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
    stop_words = stopwords
    word_tokens = word_tokenize(text) 
    
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



def pre_process(corpus, stopWords , enminiscula= True):
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
    if enminiscula:
        corpus = corpus.lower()
    stopset = stopwords+ list(string.punctuation)

    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    #corpus = unidecode.unidecode(corpus)
    return corpus

def lematizar(frase):
    '''
    Esta función recibe un string y le aplica lematización:
    [correr corrido correrá] ---> [correr, correr, correr]
    '''
    nlp = es_core_news_sm.load()
    doc = nlp(frase)
    lemmas = [tok.lemma_.lower() for tok in doc]
    return lemmas

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

def get_closest(word, n, model):
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

def calculo_similitud(word1, array_palabras, model, threshold = 0.5):
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







