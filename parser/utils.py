import re
from cts import *
import fitz
from nltk.tokenize import sent_tokenize, word_tokenize
import en_core_web_sm
from spacy.matcher import Matcher

def extract_text(path):
    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.getText()
        return text

def retrieve_email(text):
    return re.findall('\S+@\S+', text)


def retrieve_phone_number(text):
    regex = re.compile("\+?\d[\( -]?\d{3}[\) -]?\d{3}[ -]?\d{2}[ -]?\d{2}")
    numbers = re.findall(regex, text)
    #print(numbers)
    return numbers


def retrieve_skills(text):
    return [skill for skill in skills if (' '+skill.lower()+' ') in text.lower()]


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
    pass


def retrieve_last_experience_year(text):
    pass

NAME_PATTERN      = [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]
def retrieve_name(text):

    '''
    Helper function to extract name from spacy nlp text
    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param matcher: object of `spacy.matcher.Matcher`
    :return: string of full name
    '''
    nlp = en_core_web_sm.load()
    matcher = Matcher(nlp.vocab)
    pattern = [NAME_PATTERN]
    nlp_text = nlp(text)
    matcher.add('NAME', None, *pattern)
    
    matches = matcher(nlp_text)
    #print(matches)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        #print(span.text)
        return span.text


def parse_cv_sections(text):
    pass






