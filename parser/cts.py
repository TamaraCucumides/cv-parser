import pandas as pd
import os
import sys

# Archivo usado para los diccionarios constantes


def cargar_dict(path):
    '''
    Utilidad para cargar los diccionarios
    '''
    with open(path) as f:  
        array = [x.strip() for x in f]
        c = [x for x in array if x != ''] # '' aparece cuando hay lines vacias
    return c

cwd = os.getcwd() +'/parser/diccionarios/'


grados_educativos_orden = cargar_dict(cwd + 'grados_educativos_orden')
educacion = cargar_dict(cwd + 'universidades')
educacionSiglas = cargar_dict(cwd + 'universidades_siglas')
idiomas = cargar_dict(cwd + 'idiomas')
idiomas_nivel = cargar_dict(cwd + 'idiomas_nivel')
palabras_claves = cargar_dict(cwd + 'palabras_claves')
licencias = cargar_dict(cwd + 'licencias_certificaciones')


