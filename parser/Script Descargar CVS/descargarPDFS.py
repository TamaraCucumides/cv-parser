import sys
import pandas as pd
import os
import wget

numero = int(sys.argv[1])
file_name = sys.argv[2]

print('Descargando '+ str(numero) + ' CVs')

cwd = os.getcwd()

links_data = pd.read_csv(file_name, encoding = 'utf-8', header= 0)
links_data.columns = ['ID', 'NOMBRE_CV']
filas = len(links_data.index)
print(str(filas) +' CVs disponibles')

def descargarCVs(dataframe, numero):
    link_base = 'https://genoma-archives.s3.us-east-2.amazonaws.com/'
    names = dataframe.NOMBRE_CV.values[0:numero]
    for name in names:
        print(name)
        url = link_base + name
        wget.download(url,  name)
descargarCVs(links_data, numero)
