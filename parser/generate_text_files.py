import re
import fitz
import os
import spacy
import regex
import unidecode

def extract_text(path):
    '''
    Input: ruta hacia los archivos
    Salida: Texto plano como string
    '''
    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.getText()

        
        #text = re.sub(' +', ' ', text)    
        text = regex.sub("[^\P{P}-.,+@:/]+", "", text) # eliminar todas los simbolos excepto +-.,@:/
        text = text.replace('\r\n', ' ') # Trata de eliminar multiples saltos de linea.
        # Teoricamente, estos simbolos ya deberian estar eliminados, a veces aparecen igual, solo se reemplazan por si
        # aparecen de nuevo
        text = text.replace('✓', '')
        text = text.replace('|',' ')
        text = text.replace('', '')
        text = text.replace('', '')
        text = text.replace('�', '')
        text = text.replace('','')
        text = " ".join(text.split()) #Eliminacion de varios espacion "hola  como      estas" ---> "hola como estas"

        text_2 = ''
        # ANTECENTES, PEDRO, estudiando ----> Antecedentes, Pedro, estudiando
        for word in text.split():
            if word.isupper():
                text_2+=word.capitalize()+' '
            else:
                text_2 += word+ ' '
        text = text_2
        return text

if __name__ == '__main__':
    resumes = []
    for root, _, filenames in os.walk('resumes'):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)
    for resume in resumes:
        name = resume.replace("resumes/", '').replace('.pdf', '')
        text = extract_text(resume)
        text_file = open('resumes_text_output/'+name, "wt",encoding='utf-8')
        n = text_file.write(text)
        text_file.close()