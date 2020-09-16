import pandas as pd
import os
import sys
import json

# Archivo usado para los diccionarios constantes


def cargar_dict(path):
    '''
    Utilidad para cargar los diccionarios
    '''
    with open(path) as f:  
        array = [x.strip() for x in f]
        c = [x for x in array if x != ''] # '' aparece cuando hay lines vacias, las eliminamos.
    return c

def secciones_limpio(dataframe):
    ar = [str(exp).lower() for exp in dataframe if str(exp)!='nan' and str(exp)!= ' ']
    return ar

cwd = os.getcwd() +'/diccionarios/'
path_secciones_dic = os.getcwd() + '/CSVs/Secciones CVs_buscador.csv'


secciones_dic = pd.read_csv(path_secciones_dic)
experiencias = secciones_limpio(secciones_dic.Experiencia)
perfil = secciones_limpio(secciones_dic.Perfil)
educacion_sec = secciones_limpio(secciones_dic.Educacion)
cursos = secciones_limpio(secciones_dic.Cursos)
habilidades = secciones_limpio(secciones_dic.Habilidades) 
contacto = secciones_limpio(secciones_dic.Contacto)
referencias = secciones_limpio(secciones_dic.Referencias)
logros = secciones_limpio(secciones_dic.Logros)
hobbies = secciones_limpio(secciones_dic.Hobbies)

with open(cwd + '/grados_educativos_orden.json') as json_file:
        grados_educativos_orden = json.load(json_file)

#grados_educativos_orden = cargar_dict(cwd + 'grados_educativos_orden')
educacion = cargar_dict(cwd + 'universidades')
educacionSiglas = cargar_dict(cwd + 'universidades_siglas')
idiomas = cargar_dict(cwd + 'idiomas')
idiomas_nivel = cargar_dict(cwd + 'idiomas_nivel')
palabras_claves = cargar_dict(cwd + 'palabras_claves')
licencias = cargar_dict(cwd + 'licencias_certificaciones')

