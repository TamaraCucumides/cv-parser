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
import es_core_news_sm
import itertools
from nltk.stem import SnowballStemmer
import textacy
import regex
import unidecode
import numpy as np
import json
from gensim.models.keyedvectors import KeyedVectors
import pprint
wordvectors_file_vec ='/home/erwin/Genoma/cv-parser/fasttext-sbwc.3.6.e20.vec'


cantidad = 100000

model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)

path_to_json = 'output_seccionado/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
#print(json_files)  # for me this prints ['foo.json']
jsons = []
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        jsons.append(json.load(json_file))



def sent2vec(s):
    '''Generate Vectora for sentences.'''
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
    '''Return Cosine Similarity.'''
    return  np.dot(vec1,vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))

def get_closest(word, n):
    '''Get n most similar words by words.'''
    #This function can easily be expanded to get similar words to phrases--
    #using sent2vec() method defined in WithWord2Vec notebook. 
    word = word.lower()
    words = [word]
    similar_vals = [1]
    try:
        similar_list = model.most_similar(positive=[word],topn=n)
        
        for tupl in similar_list:
            words.append(tupl[0])
            similar_vals.append(tupl[1])
    except:
        #If word not in vocabulary return same word and 1 similarity-- 
        #see initialisation of words, similarities.
        pass
    
    return words, similar_vals












# Descripción del trabajo

prc_description = '''ingeniería máster python excel desarrollo  experiencia manejo 
machine learning metodologías ágiles liderar equipos planificar organizar trello electrónica informática java'''


# Expandir descripcion
word_value = {}
similar_words_needed = 2
for word in prc_description.split():
    similar_words, similarity = get_closest(word, similar_words_needed)
    for i in range(len(similar_words)):
        word_value[similar_words[i]] = word_value.get(similar_words[i], 0)+similarity[i]
        #print(similar_words[i], word_value[similar_words[i]])
        #print('------------------------------------------------')


no_of_cv = len(jsons)

count = {}
idf = {}
for word in word_value.keys():
    count[word] = 1
    for i in range(no_of_cv):
        #jsons[i]['skills'] = [x.lower() for x in jsons[i]['skills']]
        try:
            #if word in cvs.loc(0)['skill'][i].split() or word in cvs.loc(0)['exp'][i].split():
            if word in jsons[i]['skills'] or word in jsons[i]['experiencia'] or word in jsons[i]['educación']:
                #print('entre')
                count[word] += 1
        except:
            pass
    idf[word] = math.log(no_of_cv/count[word])




score = {}
for i in range(no_of_cv):
    score[i] = 0
    try:
        for word in word_value.keys():
            #tf = jsons[i]['skills'].count(word) + cvs.loc(0)['exp'][i].split().count(word)
            tf = jsons[i]['skills'].count(word) + jsons[i]['experiencia'].count(word) + jsons[i]['educación'].count(word)
            score[i] += word_value[word]*tf*idf[word]
    except:
        pass


sorted_list = []
for i in range(no_of_cv):
    sorted_list.append((score[i], jsons[i]['nombre archivo']))
    
sorted_list.sort(reverse = True)

pprint.pprint(sorted_list)








#if __name__ == '__main__':
