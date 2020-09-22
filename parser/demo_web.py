import streamlit as st
import os
from constantes import cargar_dict
import json


##import streamlit_theme as stt

##stt.set_theme({'primary': '#1b3388'})
#st.title('My themed app')
st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)

st.title("Demo CV_parser")

cwd = os.getcwd()
folder_csv = 'Outputs/output_text'



def file_selector(folder_path):
    '''
    Funcion que lista todos los paths a los archivos que se 
    encuentren un carpeta, el usuario selecciona 1 archivo.
    Input: String con el root.
    Output: string con path al arcchivo seleccionado.
    '''
    filenames = os.listdir(folder_path)
    #filenames=[]
    #for dirpath, dirnames, filenames in os.walk("."):
    #    for filename in [f for f in filenames if f.endswith(".csv")]:
    #        filenames.append(filename)

    #filenames = list(set(filenames))
    filenames.sort()
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def file_selector_filter(folder_path, skills_to_filter):
    filenames = os.listdir(folder_path)
    files_filtered = []
    #st.write(filenames)
    # ahora a cargar los json y ver si tienen los skills deseados:
    if skills_to_filter:
        for filename in filenames:
            with open('/home/erwin/Genoma/cv-parser/parser/Outputs/output_parser/' + filename, encoding='utf-8', errors='ignore') as read_file: 
                #st.write(cwd + '/'+filename)
                data = read_file
                #st.write(data.read())
                dict_cv = json.load(data)
                skills_cv = [item.lower() for item in dict_cv['Skills']]
                #for elemento in skills_cv:
                #    if elemento in skills_to_filter:
                flag = 0
                if(all(x in skills_cv for x in skills_to_filter)): 
                    flag = 1
                if flag:
                    files_filtered.append(filename)
    else:
        files_filtered = filenames

    files_filtered.sort()
    selected_filename = st.selectbox('Select a file', files_filtered)
    #st.write(files_filtered)
    return os.path.join(folder_path, selected_filename)
        

st.subheader("CV a .txt")

filename = file_selector('Outputs/output_text')

#st.write('You selected `%s`' % filename)
mostrar_cvtxt = st.checkbox('Mostrar CV_txt')


#Print csv seleccionado
if mostrar_cvtxt:
    #for line in open(filename).read().splitlines():
    #    st.text(line)
    st.text(open(filename).read())



st.subheader("CV.txt a CV parseado .json")
filename_par = file_selector(folder_path='Outputs/output_parser')
mostrar_cvparseado = st.checkbox('Mostrar CV parseado')
if mostrar_cvparseado:
    #cv = open(filename_par)
    #st.write(cv)
    with open('/home/erwin/Genoma/cv-parser/parser/' + filename_par, encoding='utf-8', errors='ignore') as read_file: 
        data = read_file
        #    st.write(filename_filtered)
        dict_cv = json.load(data)
        st.write(dict_cv)
    #for line in open(filename_par).read().splitlines():
    #    st.write(line)



st.title("Filtrar")

skills = cargar_dict(os.getcwd() +'/diccionarios/skills')

selec_skills = st.multiselect("Seleccionar skills", skills)

#st.write(selec_skills)
if selec_skills:
    try:
        filename_filtered = file_selector_filter('Outputs/output_parser', selec_skills)
        mostrar_cvtxt_fil = st.checkbox('Mostrar CV')
        
        if mostrar_cvtxt_fil:
            mostrar_solo_skills = st.checkbox('Mostrar solo skills')
            if not mostrar_solo_skills:
                with open('/home/erwin/Genoma/cv-parser/parser/' + filename_filtered, encoding='utf-8', errors='ignore') as read_file: 
                    data = read_file
                #    st.write(filename_filtered)
                    dict_cv = json.load(data)
                    st.write(dict_cv)
            else:
                #st.write('Hola')
                #st.write(filename_filtered)
                with open('/home/erwin/Genoma/cv-parser/parser/' + filename_filtered, encoding='utf-8', errors='ignore') as read_file: 
                    data = read_file
                #    st.write(filename_filtered)
                    dict_cv = json.load(data)
                    st.write(dict_cv['Skills'])

    except:
        st.write('No hay nadie que tenga todos esos skills :(')

