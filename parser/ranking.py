import re
import math
import fitz
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.matcher import Matcher
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
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
import string






wordvectors_file_vec = os.getcwd()+ '/parser/embeddings/fasttext-sbwc.3.6.e20.vec'


cantidad = 100000

model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)

path_to_json = os.getcwd() + '/parser/Outputs/output_seccionado'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
#print(json_files)  # for me this prints ['foo.json']
jsons = []
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        jsons.append(json.load(json_file))






def pre_process(corpus):
    corpus = corpus.lower()
    #stopwords_new = stopwords.words('spanish').append('experiencia')
    stopwords = nltk.corpus.stopwords.words('spanish')
    newStopWords = ['experiencia','conocimiento','desarrollo', 'ingeniero']
    stopwords.extend(newStopWords)

    stopset = stopwords+ list(string.punctuation)

    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    #corpus = unidecode.unidecode(corpus)
    return corpus

def lematizar(frase):
    nlp = es_core_news_sm.load()
    doc = nlp(frase)
    lemmas = [tok.lemma_.lower() for tok in doc]
    return lemmas






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

def calculo_similitud(word1, array_palabras):
    n_veces = 0
    for word in array_palabras:
        try:
            sim = model.similarity(word, word1)
            #print(word)
            #print(sim)
            if sim > 0.5:
                n_veces += 1
            else:
                continue
        except: #No estaba la palabra
            pass
        
    return n_veces



# Descripción del trabajo

#prc_description = '''ingeniería máster python excel desarrollo  experiencia manejo 
#machine learning metodologías ágiles liderar equipos planificar organizar trello electrónica informática java'''


file = '/home/erwin/Genoma/cv-parser/parser/Descripcion_cargo/descripcion_cargo'
with open(file) as f:
  prc_description = " ".join([x.strip() for x in f]) 

prc_description = pre_process(prc_description)
print(prc_description)

# Expandir descripcion
word_value = {}
similar_words_needed = 2
for word in prc_description.split():
    similar_words, similarity = get_closest(word, similar_words_needed)
    for i in range(len(similar_words)):
        word_value[similar_words[i]] = word_value.get(similar_words[i], 0)+similarity[i]





#no_of_cv = len(jsons)

#count = {}
#idf = {}
#for word in word_value.keys():
#    count[word] = 0
#    for i in range(no_of_cv):
#        try:
#            if word in jsons[i]['skills'] or word in jsons[i]['experiencia'] or word in jsons[i]['educación']:
#                #print('entre')
#                count[word] += 1
#        except:
#            pass
#    idf[word] = math.log((no_of_cv + 1)/(count[word]+1))+1


no_of_cv = len(jsons)

count = {}
idf = {}
for word in word_value.keys():
    count[word] = 0
    for i in range(no_of_cv):
        # eliminación de stopwords y quizas lematizacion
        skill_pro = pre_process(jsons[i]['skills']) 
        expe_pro = pre_process(jsons[i]['experiencia'])
        edu_pro = pre_process(jsons[i]['educación'])
        
        if calculo_similitud(word, skill_pro.split()) or calculo_similitud(word, expe_pro.split()) or calculo_similitud(word, edu_pro.split()):
            count[word] += 1


    idf[word] = math.log((no_of_cv+1)/(1+count[word]))








#score = {}
#for i in range(no_of_cv):
#    score[i] = 0
#    try:
#        for word in word_value.keys():
#            tf = jsons[i]['skills'].count(word) + jsons[i]['experiencia'].count(word) + jsons[i]['educación'].count(word)
#            score[i] += word_value[word]*tf*idf[word]
#    except:
#        pass



score = {}
for i in range(no_of_cv):
    score[i] = 0
    skill_pro = pre_process(jsons[i]['skills']) 
    expe_pro = pre_process(jsons[i]['experiencia'])
    edu_pro = pre_process(jsons[i]['educación'])
    for word in word_value.keys():
        
        
        
        tf = 1 + calculo_similitud(word, skill_pro.split()) + calculo_similitud(word, expe_pro.split()) + calculo_similitud(word, edu_pro.split())

        score[i] += word_value[word]*tf*idf[word]





sorted_list = []
for i in range(no_of_cv):
    sorted_list.append((score[i], jsons[i]['nombre archivo']))
    
sorted_list.sort(reverse = True)

pprint.pprint(sorted_list)

