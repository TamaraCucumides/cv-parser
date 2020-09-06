import re
from cts import *
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
import unidecode


def extract_text(path):
    '''
    Input: ruta hacia los archivos
    Salida: Texto plano como string
    '''
    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.getText()
        # eliminar estos simbolos
        simbolos = '\n ,./:@'
        text_clean = ''
        for char in text:
            if (char.isalnum())| (char in simbolos):
                text_clean += char
    
        # Limpiar palabras completamente en mayusculas, es importante
        # para reconocer los nombres
        text_2 = ''
        for line in text_clean.splitlines():
            if not line.strip(): #si la linea esta vacia, saltar
                continue
            line_2=''
            for word in line.split():
                if word.isupper():
                    line_2 += word.capitalize()+' '

                else:
                    line_2 += word+ ' '
            text_2 += line_2 +'\n'
        text = text_2
    return text

def retrieve_email(text):
    '''
    Input: Recibe texto plano
    Output: String que representa un mail.
    '''
    mails = re.findall('\S+@\S+', text)
    if len(mails)>1:
        mails  =mails[0]
    return mails


def retrieve_phone_number(text):
    '''
    Retorna numero de 12 digitos
    que parten con +
    Input: Texto plano
    Output: Texto plano
    '''
    regex = re.compile("\+?\d[\( -]?\d{3}[\) -]?\d{3}[ -]?\d{2}[ -]?\d{2}")
    regex = re.compile("\d{8,11}")
    texto_busqueda = "".join(text.split()) 
    numbers = re.findall(regex, texto_busqueda)
    if len(numbers)>1:
        numbers = numbers[0]

    return numbers


def retrieve_skills(nlp_text):
    '''
    Funcion que buisca los skill declarados del postulante
    Se buscan tanto skill de 1 token como de varios.
    '''
    tokens = [token.text for token in nlp_text if not token.is_stop]
    data = pd.read_csv(os.path.join(os.getcwd(), 'parser/skills.csv')) 
    skills = list(data.columns.values)

    skillset = []
    noun_chunks = list(nlp_text.noun_chunks)


    # check for one-grams
    for token in tokens:
        token_un = unidecode.unidecode(token)
        if token_un.lower() in skills:
            skillset.append(token)
    
    # check for bi-grams and tri-grams
    for chunk in noun_chunks:
        st = chunk.text
        chunk_lower = " ".join(st.split()).lower()       
        for skill in skills:
                skill_un = unidecode.unidecode(skill)
                chunk_un = unidecode.unidecode(chunk_lower)
                if skill_un in chunk_un:
                    skillset.append(skill.capitalize())
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def retrieve_education_institution(text, nlp_text):
    '''
    Funcion que recupera las universidad o intituciones mencionadas en el CV
    Hace uso de 2 listas: Educacion y Educacion_siglas.
    Input: texto plano
    Output: Lista de strings unicos
    '''
   
    educacion_list=[]

    
    filter_noun = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text)) if pos[0] == 'N']   
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

    unique_values = set(educacion_list)

    return list(unique_values) 
 

def retrieve_languages(text, nlp_text):
    '''
    Funcion que recupera los idiomas que declara el postulante.
    '''
    nlp = es_core_news_sm.load()
    nlp_text = nlp(re.sub(r'[^\w\s]','',text))
    combinaciones = list(itertools.product(idiomas, idiomas_nivel))
    combinaciones_strings = []
    for i in range(1, len(combinaciones)):
        combinaciones_strings.append(combinaciones[i][0] + combinaciones[i][1])
    combinaciones_strings = combinaciones_strings + idiomas

    noun_chunks = list(nlp_text.noun_chunks)
    
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
    frases = sent_tokenize(text)
    #for frase in frases:
    for grado in grados_educativos_orden:
         for frase in frases:
            grado_un = unidecode.unidecode(grado)
            frase_un = unidecode.unidecode(frase)
            if grado_un.lower() in frase_un.lower():
                education.append(grado.capitalize())

    if len(education)>0:
        education = [education[-1]]
    else:
        education = []

    return education


def retrieve_dates(text):
    pass


def retrieve_past_experience(text):
    
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




def retrieve_experience_2(text):
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
    


def retrieve_last_experience_year(text):
    pass


def delete_words(text):

    text = text.replace('\n', ' ')
    # Esto es hardcoding, pero he intentado todo y no me resulta sin este trucazo
    # Siempre se colan estas palabras en la deteccion de nombres
    text = text.replace("Pontificia", " ")
    text = text.replace("Universidad", " ")
    #text = text.replace("PONTIFICIA", " ")
    #text = text.replace("UNIVERSIDAD", " ")
    text = text.replace("Curriculum", " ")
    #text = text.replace("CURRILUM", " ")
    text = text.replace("Vitae", " ")
    #text = text.replace("VITAE", " ")
    #text = text.replace("Calle", " ")
    #text = text.replace("•", " ")
    #text = text.replace("▪", " ")
    #text = text.replace("-", " ")

    return text


def retrieve_name(text, nlp_text):

    '''
    Funcion que busca por 3 pronombres seguidos. Se recibe el texto
    plano y se procesa para sacar simbolos y palabras que complican el analisis
    Input: texto plano
    Output: texto plano
    '''
    text = text[0:math.floor(len(text)/16)]
    text = delete_words(text)
    # El uso de mayusculas es importante para el matcher,
    # por ejemplo FELIPE no se reconoce, pero Felipe sí
#    string = ''
#    for word in text.split(): 
#        if word.isupper(): #FELIPE ---> Felipe, Paulina ---> Paulina
#            string += word.capitalize() +' '  
#        else:
#            string += word +' '  

 #   text = string
    NAME_PATTERN      = [{'POS': 'PROPN'}, {'POS': 'PROPN'},{'POS': 'PROPN'}]
    nlp = es_core_news_sm.load()
    #nlp_text = nlp(text.replace('@', '\n').replace('www','\n'))
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


def parse_cv_sections(text):
    pass


def summarize_cv(text, nlp_text):
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
    word_list = cv.get_feature_names();    
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
    '''
    regex = re.compile("(?:https?:)?\/\/(?:[\w]+\.)?linkedin\.com\/in\/(?P<permalink>[\w\-\_À-ÿ%]+)\/?")
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
   # stemmed_claves = [stemmer.stem(token) for token in palabras_claves]
    stop_words = set(stopwords.words('spanish')) 
    
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





