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
    regex = re.compile("\+?\d[\( -]?\d{3}[\) -]?\d{3}[ -]?\d{2}[ -]?\d{2}")
    numbers = re.findall(regex, text)
    #print(numbers)
    return numbers


def retrieve_skills(text):
    #return [skill for skill in skills if (' '+skill.lower()+' ') in text.lower()]
    nlp = en_core_web_sm.load()
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
    #tokens = word_tokenize(text)
    #print(tokens)
    educacion_list=[]
    for item in educacion:
      if item.lower() in text.lower():
          #print ('found one of em')
          educacion_list.append(item)
    return educacion_list



def retrieve_higher_degree(text):
    pass


def retrieve_dates(text):
    pass


def retrieve_past_experience(text):
    
    '''
    Helper function to extract experience from resume text
    :param resume_text: Plain resume text
    :return: list of experience
    '''
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('spanish'))
    #print(stop_words)

    # word tokenization 
    word_tokens = nltk.word_tokenize(text)

    # remove stop words and lemmatize  
    filtered_sentence = [w for w in word_tokens if not w in stop_words and wordnet_lemmatizer.lemmatize(w) not in stop_words] 
    sent = nltk.pos_tag(filtered_sentence)
    #print(sent)
    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)
    #print(cs)
    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)
    
    test = []
    
    for vp in list(cs.subtrees(filter=lambda x: x.label()=='P')):
        test.append(" ".join([i[0] for i in vp.leaves() if len(vp.leaves()) >= 2]))

    # Search the word 'experience' in the chunk and then print out the text after it
    x = [x[x.lower().index('laboral') + 10:] for i, x in enumerate(test) if x and 'laboral' in x.lower()]
    return x
    


def retrieve_last_experience_year(text):
    pass


def retrieve_name(text):

    '''
    busca por 3 pronombres seguidos, se cae cuando alguien pone sus cuatro nombres
    '''
    NAME_PATTERN      = [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]
    nlp = en_core_web_sm.load()
    matcher = Matcher(nlp.vocab)
    pattern = [NAME_PATTERN]
    nlp_text = nlp(text)
    matcher.add('NAME', None, *pattern)
    
    matches = matcher(nlp_text)
    #print(matches)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text


def parse_cv_sections(text):
    pass






