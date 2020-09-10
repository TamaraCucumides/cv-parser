import re
import nltk
import pandas as pd
import os
import es_core_news_md
import itertools
from gensim.models.keyedvectors import KeyedVectors
import re
import json
import pprint
pp = pprint.PrettyPrinter(indent=4)
import multiprocessing as mp
from utils import pre_process

# Utilidad para borrar simbolos
re_c = re.compile(r'\w+')

# Se carga el modelo español de spacy
nlp = es_core_news_md.load()

# Se carga el modelo de embeddings en español
print("Cargando embeddings")
wordvectors_file_vec = os.getcwd() + '/parser/embeddings/fasttext-sbwc.3.6.e20.vec'
cantidad = 100000
model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)
print("Embeddings cargadas")

#Se carga la tabla de secciones.
seccion_csv = os.getcwd() +'/parser/CSVs/seccionesCV.csv'
secciones = pd.read_csv(seccion_csv, header = 0)


# Se carga el dccionario de secciones, se considera todas las celdas de seccion_csv que no sean nan.
secciones_dict = {
    'extras' : [str(x.lower()) for x in secciones.Perfil.values if str(x)!= 'nan'],
    'experiencia' : [str(x.lower()) for x in secciones['Experiencia '].values if str(x)!= 'nan'],
    'educación' : [str(x.lower()) for x in secciones['Formación Académica'].values if str(x)!= 'nan'],
    'skills' : [str(x.lower()) for x in secciones['skills'].values if str(x)!= 'nan']                   
        
}



#umbral para similitud de secciones.
threshold = 0.5


similar_to = secciones_dict





lista_secciones = secciones_dict.keys()

# Usando secciones_dict que tiene las secciones a buscar y palabras que describen esas secciones
# se llevan aquellas palabras a su lema
for section in lista_secciones:
    new_list = []    
    for word in similar_to[section]:
        docx = nlp(word)
        new_list.append(docx[0].lemma_)
    
        
    similar_to[section] = list(set(new_list)) # se retorna lista de elementos unicos



def modificar(word):
    '''
     Retornar el lema de la palabra. 
   Fijandose que no sean un simbolo raro ni una stopword.
    '''
    try:
        symbols = '''~`!@#$%^&*)(_+-=}{][|\:;",./<>?'''
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
    

    

def esta_vacia(line):
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
    seccion_previa  = 'extras'

    for line in cv_txt:
        # si la linea esta vacia, entonces saltar
        if (len(line.strip()) == 0 or esta_vacia(line)):
            continue

        # procesar la linea
        palabras_en_linea = re_c.findall(line)
        lista_palabras_utiles  = []
        
        # recorrer todas las palabras de linea actual
        for i in range(len(palabras_en_linea)):
            palabra_limpia = modificar(palabras_en_linea[i])

            if (palabra_limpia): 
                lista_palabras_utiles.append(palabra_limpia)

        linea_actual = ' '.join(lista_palabras_utiles) # crear un string a partir de una lista
        doc = nlp(linea_actual)
        valor_seccion = {}

        # Inicializar el valor de la seccion a 0
        for section in lista_secciones:
            valor_seccion[section] = 0.0
        valor_seccion[None] = 0.0

        # Actualizar los valores de las secciones   
        for token in doc:
            for section in lista_secciones:
                for word in similar_to[section]:
                    try:
                        valor_seccion[section] = max(valor_seccion[section], float(model.similarity(token.text, word)))
                    except: 
                        pass # si es que token.text no esta en el vocabulario
                    
        # ver la siguiente sección probable de acuerdo al umbral establecido
        siguiente_seccion_prob = None
        for section in lista_secciones:
            if (valor_seccion[siguiente_seccion_prob] < valor_seccion[section] and valor_seccion[section] > threshold):
                siguiente_seccion_prob = section

        # Actualizar la seccion previa
        if (seccion_previa != siguiente_seccion_prob and siguiente_seccion_prob is not None):
            seccion_previa = siguiente_seccion_prob


        # Concatenar las palabras para formar un linea, las palabras se usan en su lema
        try:
            docx = nlp(line)
        except:
            continue  # si que hay simbolos raros
        linea_lematizada = ''
        for token in docx:
            if (not token.is_stop):
                linea_lematizada += token.lemma_ + ' '

        secciones_data[seccion_previa] += pre_process(linea_lematizada)+ ' ' # se eliman los saltos de linea.


    cv_txt.close()
    return secciones_data




def generate_json(cv):
    name = cv.replace(direc + dir_txt, '')        
    secciones_data = seccionar_cv(cv)
    secciones_data['nombre archivo'] = name

    with open(direc + dir_output + name+'.json', 'w',encoding='utf-8') as json_file:
        json.dump(secciones_data, json_file,ensure_ascii=False, indent=4)


    


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    #print('Usando ' + str(mp.cpu_count()) + ' cores')
    direc = os.getcwd()
    dir_txt = '/parser/Outputs/output_text/'
    dir_output = '/parser/Outputs/output_seccionado/'


    # Se cargan todos los paths a los cv en formato .txt 
    resumes_seccionado = []
    for root, _, filenames in os.walk(direc + dir_txt):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes_seccionado.append(file)

    print("Seccionando CVs: "+str(len(resumes_seccionado)))

    # Se secciona cada CV
    seccionados = [pool.apply_async(generate_json(cv)) for cv in resumes_seccionado]
    pool.close()
    pool.join()      

    print('Finalizado')





  