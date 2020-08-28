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
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import es_core_news_sm
import itertools
from nltk.stem import SnowballStemmer

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
    texto_busqueda = "".join(text.split()) 
    numbers = re.findall(regex, texto_busqueda)
    if len(numbers)>1:
        numbers = numbers[0]

    return numbers


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
 

def retrieve_languages(text):
    combinaciones = list(itertools.product(idiomas, idiomas_nivel))
    combinaciones_strings = []
    for i in range(1, len(combinaciones)):
        combinaciones_strings.append(combinaciones[i][0] + combinaciones[i][1])
    combinaciones_strings = combinaciones_strings + idiomas
    nlp = es_core_news_sm.load()
    sr = stopwords.words('spanish')
    #educacion_list=[]
    nlp_text = nlp(text)
    
    #filter_noun = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text)) if pos[0] == 'N']   
    noun_chunks = list(nlp_text.noun_chunks)
    #print(noun_chunks)
    
    idiomas_cv = []
    for item in combinaciones_strings:
        for noun in noun_chunks:
            #print ({item.lower(): noun.text.lower()})
            if item.lower()== noun.text.lower():
                idiomas_cv.append(noun.text)
                #print(item)

    return idiomas_cv


    




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
                education.append(grado.capitalize())

    if len(education)>0:
        education = education[-1]
    else:
        education = None

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
    nlp = es_core_news_sm.load()
    matcher = Matcher(nlp.vocab)
    pattern = [NAME_PATTERN]
    nlp_text = nlp(text)
    matcher.add('NAME', None, *pattern)
    
    matches = matcher(nlp_text)
   
    for _, start, end in matches:
        span = nlp_text[start:end]
        return span.text


def parse_cv_sections(text):
    pass


def summarize_cv(text):
    '''
    Funcion que que rankea frases a partir de frecuencia
    de palabras, es un intento simple/ no muy efectivo de resumir.
    El problema de los cv es que el texto es reducido, no hablamos de un libro.

    Input: texto plano
    Output: texto plano
    '''
    nlp = es_core_news_sm.load()
    doc = nlp(text)
    corpus = [sent.text.lower() for sent in doc.sents ]
    STOP_WORDS = set(stopwords.words("spanish"))
    cv = CountVectorizer(stop_words=list(STOP_WORDS))   
    cv_fit=cv.fit_transform(corpus)    
    word_list = cv.get_feature_names();    
    count_list = cv_fit.toarray().sum(axis=0)
    word_frequency = dict(zip(word_list,count_list))
    val=sorted(word_frequency.values())
    #higher_word_frequencies = [word for word,freq in word_frequency.items() if freq in val[-3:]]
    #print("\nWords with higher frequencies: ", higher_word_frequencies)

    # gets relative frequencies of words
    higher_frequency = val[-1]

    for word in word_frequency.keys():  
        word_frequency[word] = (word_frequency[word]/higher_frequency)
        
    sentence_rank={}
    for sent in doc.sents:
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
    regex = re.compile("(?:https?:)?\/\/(?:[\w]+\.)?linkedin\.com\/in\/(?P<permalink>[\w\-\_À-ÿ%]+)\/?")
    #texto_busqueda = "".join(text.split()) 
    profile = re.findall(regex, text)
    if profile:
        return 'https://www.linkedin.com/in/' + profile[0]
    else:
        return ''


def busqueda_palabras_claves(text):
    nlp = es_core_news_sm.load()
    stemmer = SnowballStemmer('spanish')

    stemmed_claves = [stemmer.stem(token) for token in palabras_claves]
    
    stop_words = set(stopwords.words('spanish')) 
    
    word_tokens = word_tokenize(text) 
    
    filtered_text = [w for w in word_tokens if not w in stop_words] 

    encontradas = []
    for palabra_clave in stemmed_claves:
        for word in filtered_text:
            if stemmer.stem(word).lower() == palabra_clave.lower():
                encontradas.append(word.capitalize())
    encontradas = set(encontradas)

    return encontradas





