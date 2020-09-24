import streamlit as st
import os
from constantes import cargar_dict, educacion, grados_educativos_orden, licencias
import json
import unidecode as u
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords 
from utils import palabras_cercanas
import matplotlib.pyplot as plt
import nltk
import json
from wordcloud import WordCloud 
from collections import Counter
from utils import lematizar, stemizar, preprocesar_texto
from constantes import cargar_dict
import re

st.set_option('deprecation.showPyplotGlobalUse', False)
def show_words(file):
    #st.write(file)
    with open(file, "r") as read_file:
        cv_txt = json.load(read_file)
    for key in cv_txt.keys():
        if cv_txt[key] is None:
            cv_txt[key] = []
    list_words = cv_txt['PALABRAS_CLAVES']*2 + cv_txt["LICENCIA_CERTIFICACION"]*5  + cv_txt["IDIOMAS"]*2 + cv_txt["SKILLS"]*10 + cv_txt["EDUCACION"]["GRADO_MAS_ALTO"]*4
    word_could_dict=Counter(list_words)
    wordcloud = WordCloud(background_color='white',width = 500, height = 250, max_words= 50).generate_from_frequencies(word_could_dict)
    plt.figure(figsize=(10,8))
    fig = plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    st.pyplot()
    #plt.close(fig)


#wordvectors_file_vec = os.getcwd() + '/embeddings/fasttext-sbwc.3.6.e20.vec'
#cantidad = 300000
#model = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)
#model.init_sims(replace=True)
#model.save('bio_word')
#model = KeyedVectors.load('bio_word', mmap='r')

model = KeyedVectors.load('bio_word', mmap='r')

st.title("Demo CV_parser")

cwd = os.getcwd()
folder_csv = 'Outputs/output_text'


def load_dict(path, filename):
    dir_file = os.path.join(path, filename)
    #st.write(dir_file)
    with open(dir_file, encoding='utf-8', errors='ignore') as read_file: 
        data = read_file
        dict_cv = json.load(data)
    return dict_cv

def file_selector(folder_path):
    '''
    Funcion que lista todos los paths a los archivos que se 
    encuentren un carpeta, el usuario selecciona 1 archivo.
    Input: String con el root.
    Output: string con path al arcchivo seleccionado.
    '''
    filenames = os.listdir(folder_path)
    filenames.sort()
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def file_selector_filter(folder_path, skills_to_filter):
    filenames = os.listdir(folder_path)
    files_filtered = []

    # ahora a cargar los json y ver si tienen los skills deseados:
    if skills_to_filter:
        for filename in filenames:
            #st.write(filename)
            dict_cv = load_dict('/home/erwin/Genoma/cv-parser/parser/Outputs/output_parser', filename)
            skills_cv = [item.lower() for item in dict_cv['SKILLS'] + dict_cv['LICENCIA_CERTIFICACION']]
            flag = 0
            if(all(x in skills_cv for x in skills_to_filter)): 
                flag = 1
            if flag:
                files_filtered.append(filename)
    else:
        files_filtered = filenames

    files_filtered.sort()
    selected_filename = st.selectbox('Select a file', files_filtered)

    return os.path.join(folder_path, selected_filename)


def file_selector_educacion(folder_path, universidades_selected, grado):
    filenames = os.listdir(folder_path)
    files_filtered = []

    # ahora a cargar los json y ver si tienen los skills deseados:
    if universidades_selected:
        for filename in filenames:
            dict_cv = load_dict('/home/erwin/Genoma/cv-parser/parser/Outputs/output_parser', filename)
            #st.write(dict_cv['Educacion']['Universidades:'])
            Ues_cv = [u.unidecode(item.lower()) for item in dict_cv['EDUCACION']['UNIVERSIDADES']]
            flag = 0
            U_dict = [u.unidecode(i.lower()) for i in universidades_selected]
            if( U_dict[0] in Ues_cv): 
                flag = 1
            if flag:
                files_filtered.append(filename)
    else:
        files_filtered = filenames

    files_filtered.sort()
    selected_filename = st.selectbox('Select a file', files_filtered)
    if selected_filename:
        return os.path.join(folder_path, selected_filename)
        
def file_selector_grado(folder_path, universidades_selected, grado):
    filenames = os.listdir(folder_path)
    files_filtered = []

    # ahora a cargar los json y ver si tienen los skills deseados:
    if universidades_selected:
        for filename in filenames:
            dict_cv = load_dict('/home/erwin/Genoma/cv-parser/parser/Outputs/output_parser', filename)
            #st.write(dict_cv['Educacion']['Universidades:'])
            Ues_cv = [u.unidecode(item.lower()) for item in dict_cv['EDUCACION']['GRADO_MAS_ALTO']]
            flag = 0
            U_dict = [u.unidecode(i.lower()) for i in universidades_selected]
            if( U_dict[0] in Ues_cv): 
                flag = 1
            if flag:
                files_filtered.append(filename)
    else:
        files_filtered = filenames

    files_filtered.sort()
    selected_filename = st.selectbox('Select a file', files_filtered)
    if selected_filename:
        return os.path.join(folder_path, selected_filename)
st.subheader("CV en texto plano (.txt)")

filename = file_selector('Outputs/output_text')


mostrar_cvtxt = st.checkbox('Mostrar Texto')


#Print csv seleccionado
if mostrar_cvtxt:
    st.text(open(filename).read())




st.subheader(" CV parseado (.json)")
filename_par = file_selector(folder_path='Outputs/output_parser')
mostrar_cvparseado = st.checkbox('Mostrar .json')
if mostrar_cvparseado:
    dict_cv = load_dict('/home/erwin/Genoma/cv-parser/parser', filename_par)
    st.write(dict_cv)

if st.checkbox('Mostrar Nube Palabras'):
    show_words(os.path.join('/home/erwin/Genoma/cv-parser/parser', filename_par))



st.title("Filtrar")

skills = cargar_dict(os.getcwd() +'/diccionarios/skills')

selec_skills = st.multiselect("Seleccionar skills o licencia", skills + licencias )

#st.write(selec_skills)
if selec_skills:
    #try:
    filename_filtered = file_selector_filter('Outputs/output_parser', selec_skills)
    mostrar_cvtxt_fil = st.checkbox('Mostrar CV')
    
    if mostrar_cvtxt_fil:
        mostrar_solo_skills = st.checkbox('Mostrar solo skills')
        if not mostrar_solo_skills:
            dict_cv = load_dict('/home/erwin/Genoma/cv-parser/parser',filename_filtered)
            st.write(dict_cv)
        else:
            dict_cv = load_dict('/home/erwin/Genoma/cv-parser/parser',filename_filtered)
            st.write(dict_cv['Skills'])
            st.write(dict_cv['Licencias-Certificaciones'])

  


selec_u = st.multiselect("Seleccionar Universidades", educacion)
if selec_u:

    filename_filtered = file_selector_educacion('Outputs/output_parser', selec_u, 'hola')
    #st.write(filename_filtered)
    mostrar_cvtxt_fil_2 = st.checkbox('Mostrar CV U')

    if mostrar_cvtxt_fil_2:
        mostrar_solo_u = st.checkbox('Mostrar solo Educacion')
        if not mostrar_solo_u:
            dict_cv = load_dict('/home/erwin/Genoma/cv-parser/parser',filename_filtered)
            st.write(dict_cv)
        else:
            dict_cv = load_dict('/home/erwin/Genoma/cv-parser/parser',filename_filtered)
            st.write(dict_cv['Educacion'])


selec_grado = st.multiselect("Seleccionar Grado", list(grados_educativos_orden.keys()))
if selec_grado:

    filename_filtered = file_selector_grado('Outputs/output_parser', selec_grado, 'hola')
    #st.write(filename_filtered)
    mostrar_cvtxt_fil_g = st.checkbox('Mostrar CV G')

    if mostrar_cvtxt_fil_g:
        mostrar_solo_g = st.checkbox('Mostrar solo Grado')
        if not mostrar_solo_g:
            dict_cv = load_dict('/home/erwin/Genoma/cv-parser/parser',filename_filtered)
            st.write(dict_cv)
        else:
            dict_cv = load_dict('/home/erwin/Genoma/cv-parser/parser',filename_filtered)
            st.write(dict_cv['Educacion'])

st.title('Prueba de embeddings')
#cargar_embedding()


st.subheader("Ver vector")
see_vector = st.text_input("Palabra", "")
if see_vector!="":
    st.write(model[see_vector])

st.subheader('Similitud de palabras')
user_input_1 = st.text_input("Palabra 1", "")
user_input_2 = st.text_input("Palabra 2", "")
# st.write("Cargando embeddings")
#@st.cache

#st.write("Embeddings cargadas" + '\n')

if user_input_1!="" and user_input_2!="":
    #st.write("Espera un poco")
    #model = cargar_embedding()
    sim = model.similarity(user_input_1, user_input_2)
    st.write(sim)
    if sim>0.5:
        st.write('Las dos palabras pertencen a categorias similares :smile:')
    else:
        st.write('Las dos palabras no son parecidas 😑')
#        
# """
st.subheader('Palabras más similares')
similares = st.text_input("Similares a:", "")

if similares != "":
    #model = cargar_embedding()
    n=st.slider('Nmero similares', min_value=2, max_value=10, value=None, step=1, format=None, key=None)
    sim = palabras_cercanas(similares, n, model)
    if sim:
        st.write(sim)
    else:
        st.write("La palabra no esta en el vocabulario")

st.title("Extras")
st.subheader('Lematizador')
lemas = st.text_input("Palabra a lematizar:", "")

if lemas != "":
    #model = cargar_embedding()
    n=5
    lematizado = lematizar(lemas)
    if lematizado:
        st.write(lematizado)
    else:
        st.write("La palabra no esta en el vocabulario")


st.subheader('Stemming')
stems = st.text_input("Palabra para aplicar stemming:", "")

if stems != "":
    stem = stemizar(stems)
    if stem:
        st.write(stem)
    else:
        st.write("La palabra no esta en el vocabulario")

st.subheader('Preprocesamiento (Bye Stopwords)')
text_to_clean = st.text_input("Texto a limpiar:", "")


minus = st.checkbox('Normalizar en miniscula')
punt = st.checkbox('Eliminar puntuación')
num = st.checkbox('Eliminar números')
simb = st.checkbox('Eliminar símbolos')
sto = st.checkbox('Eliminar palabras vacías')
lemm = st.checkbox('Lematizar')


if text_to_clean != "":
    st.write(text_to_clean)
    if sto:
        newStopWords = cargar_dict(os.getcwd() + '/diccionarios/stop_words_nombres')
        stopwords = nltk.corpus.stopwords.words('spanish')
        stopwords.extend(newStopWords)
    else:
        stopwords = None
    clean = preprocesar_texto(text_to_clean,stopwords , enminiscula=  minus, keepPuntuacion = not punt, keepNumeros = not num)
    if clean and not simb:
        if lemm:
            clean = lematizar(clean)
        else:  
            st.markdown('**'+clean+'**')
    elif clean and simb:
        if lemm:
            clean = lematizar(clean)
            clean = re.sub(r'[^\w\s]',' ',clean) #eliminar puntuacion
            st.markdown('**'+clean+'**')
        else:
            clean = re.sub(r'[^\w\s]',' ',clean) #eliminar puntuacion
            st.markdown('**'+clean+'**')

