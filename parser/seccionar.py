import re
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
import es_core_news_md
import itertools
from nltk.stem import SnowballStemmer
import textacy
import regex
import unidecode
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import re
import json
import pprint
pp = pprint.PrettyPrinter(indent=4)

re_c = re.compile(r'\w+')
wordvectors_file_vec = os.getcwd() + '/parser/embeddings/fasttext-sbwc.3.6.e20.vec'
nlp = es_core_news_md.load()
cantidad = 100000

model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)






seccion_csv = os.getcwd() +'/parser/CSVs/seccionesCV.csv'
#print(seccion_csv)
secciones = pd.read_csv(seccion_csv, header = 0)
#secciones.columns = secciones.loc[0] 
#secciones.columns


secciones_dict = {
    'extras' : [str(x.lower()) for x in secciones.Perfil.values if str(x)!= 'nan'],
    'experiencia' : [str(x.lower()) for x in secciones['Experiencia '].values if str(x)!= 'nan'],
    'educación' : [str(x.lower()) for x in secciones['Formación Académica'].values if str(x)!= 'nan'],
    'skills' : [str(x.lower()) for x in secciones['skills'].values if str(x)!= 'nan']                   
        
}



# switch for debug
flag_print = False

# switch to clear existing data
flag_clear = True

#threshold value for determining section
threshold = 0.45

similar_to = secciones_dict


list_of_sections = similar_to.keys()

# Usando secciones_dict que tiene las secciones a buscar y palabras que describen esas secciones
# se llevan aquellas palabras a su lema
for section in list_of_sections:
    new_list = []
    
    for word in similar_to[section]:
        docx = nlp(word)
        new_list.append(docx[0].lemma_)
    
        
    similar_to[section] = list(set(new_list)) # se retorna lista de elementos unicos
#pp.pprint(similar_to)


# function to remove unnecessary symbols and stopwords 
def modify(word):
    try:
        symbols = '''~'`!@#$%^&*)(_+-=}{][|\:;",./<>?'''
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
    

    

def is_empty(line):
    '''
    Retorna un booleano correspondiendo a 
    si una linea esta vacia en términos de letras-números
    '''
    for c in line:
        if (c.isalpha()):
            return False
    return True


def seccionar_cv(path):
    # Se crea un diccionario vacio para rellenarlo
    secciones_data = {
        'nombre archivo':'',
        'extras' : '',
        'experiencia' : '',
        'educación' : '',
        'skills':''
                        
            
    }
    # Se carga un archivo .txt, que contiene el CV que venia del PDF

    file = path
    cv_txt = open(file, "r")
    previous_section  = 'extras'

    for line in cv_txt:
        # si la linea esta vacia, entonces saltar
        if (len(line.strip()) == 0 or is_empty(line)):
            continue

        # procesar la siguiente linea
        list_of_words_in_line = re_c.findall(line)
        list_of_imp_words_in_line  = []
        
        # recorrer todas las palabras de linea actual
        for i in range(len(list_of_words_in_line)):
            modified_word = modify(list_of_words_in_line[i])

            if (modified_word): 
                list_of_imp_words_in_line.append(modified_word)

        curr_line = ' '.join(list_of_imp_words_in_line)
        doc = nlp(curr_line)
        #print(doc)
        section_value = {}

        # initializing section values to zero
        for section in list_of_sections:
            section_value[section] = 0.0
        section_value[None] = 0.0

        # updating section values    
        for token in doc:
            for section in list_of_sections:
                for word in similar_to[section]:
                    #word_token = doc.vocab[word]
                    try:
                        section_value[section] = max(section_value[section], float(model.similarity(token.text, word)))
                    except: 
                        pass # si es que token.text no esta en el vocabulario
                    
        # ver la siguiente sección de acuerdo al umbral establecido
        most_likely_section = None
        for section in list_of_sections:
            if (section_value[most_likely_section] < section_value[section] and section_value[section] > threshold):
                most_likely_section = section

        # updating the section
        if (previous_section != most_likely_section and most_likely_section is not None):
            previous_section = most_likely_section


        # writing data to the pandas series
        try:
            docx = nlp(line)
        except:
            continue  # si que hay simbolos raros
        mod_line = ''
        for token in docx:
            if (not token.is_stop):
                mod_line += token.lemma_ + ' '

        secciones_data[previous_section] += mod_line.lower()


    cv_txt.close()
    return secciones_data
    


if __name__ == '__main__':

    direc = os.getcwd()
    dir_txt = '/parser/Outputs/output_text/'
    dir_output = '/parser/Outputs/output_seccionado/'



    resumes_seccionado = []
    for root, _, filenames in os.walk(direc + dir_txt):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes_seccionado.append(file)

    print("Seccionando CVs: "+str(len(resumes_seccionado)))
    for cv in resumes_seccionado:
        name = cv.replace(direc + dir_txt, '')
        
        secciones_data = seccionar_cv(cv)
        secciones_data['nombre archivo'] = name
        with open(direc + dir_output + name+'.json', 'w',encoding='utf-8') as json_file:
            json.dump(secciones_data, json_file,ensure_ascii=False, indent=4)  
    print('Finalizado')





  