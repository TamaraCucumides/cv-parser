import re
from cts import *
import fitz
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import en_core_web_sm
from spacy.matcher import Matcher
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
import es_core_news_sm
from nltk.tokenize import sent_tokenize
import os

def extract_text(path):
    '''
    Input: ruta hacia los archivos
    Salida: Texto plano como string
    '''
    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.getText()
        return text

def retrieve_email(text):
    '''
    Input: Recibe texto plano
    Output: String que representa un mail.
    '''
    return re.findall('\S+@\S+', text)


def retrieve_phone_number(text):
    '''
    Funciona con numero del estilo : +56986634232
    se cae con +569 86634232, el espacio mata el RegExr
    '''
    regex = re.compile("\+?\d[\( -]?\d{3}[\) -]?\d{3}[ -]?\d{2}[ -]?\d{2}")
    numbers = re.findall(regex, text)
    #print(numbers)
    return numbers
"""     custom_regex = None
    if not custom_regex:
        mob_num_regex = r'''(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)
                        [-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'''
        phone = re.findall(re.compile(mob_num_regex), text)
    else:
        phone = re.findall(re.compile(custom_regex), text)
    if phone:
        number = ''.join(phone[0])
        return number
 """

def retrieve_skills(text):
    #return [skill for skill in skills if (' '+skill.lower()+' ') in text.lower()]
    nlp = es_core_news_sm.load()
    nlp_text = nlp(text)
    tokens = [token.text for token in nlp_text if not token.is_stop]
    data = pd.read_csv(os.path.join(os.getcwd(), 'parser/skills.csv')) 
    skills = list(data.columns.values)
    skillset = []
    noun_chunks = list(nlp_text.noun_chunks)
    #print(noun_chunks)
    # check for one-grams
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)
    
    # check for bi-grams and tri-grams
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def retrieve_education_institution(text):
    '''
    Funcion que recupera las universidad o intituciones mencionadas en el CV
    Hace uso de 2 listas: Educacion y Educacion_siglas.
    Input: texto plano
    Output: Lista de strings unicos
    '''
    nlp = es_core_news_sm.load()
    sr = stopwords.words('spanish')
    educacion_list=[]
    nlp_text = nlp(text)
    
    filter_noun = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text)) if pos[0] == 'N']   
    noun_chunks = list(nlp_text.noun_chunks)
    
    for item in educacion:
        for noun in noun_chunks:
            if item.lower() in noun.text.lower():
                educacion_list.append(item)
                
    for item in educacionSiglas:
        if item in filter_noun:
            educacion_list.append(item)

    unique_values = set(educacion_list)

    return list(unique_values) 
 




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
            if grado.lower() in frase.lower():
                education.append(grado)

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
    return x
    


def retrieve_last_experience_year(text):
    pass


def retrieve_name(text):

    '''
    Funcion que busca por 3 pronombres seguidos, se cae cuando alguien pone sus cuatro nombres.
    Input: texto plano
    Output: texto plano
    '''
    NAME_PATTERN      = [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]
    nlp = en_core_web_sm.load()
    matcher = Matcher(nlp.vocab)
    pattern = [NAME_PATTERN]
    nlp_text = nlp(text)
    matcher.add('NAME', None, *pattern)
    
    matches = matcher(nlp_text)
    #print(matches)
    
    for _, start, end in matches:
        span = nlp_text[start:end]
        return span.text


def parse_cv_sections(text):
    pass






