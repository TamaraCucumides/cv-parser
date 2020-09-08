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
        # eliminar estos simbolos
        
        text_clean = text
        text_2 = ''
        for line in text_clean.splitlines():
            if not line.strip(): #si la linea esta vacia, saltar
                continue
            line_2=''
            for word in line.split():
                if word.isupper():
                    line_2 += word.capitalize()+' '

                else:
                    line_2 += word+ ' '
            text_2 += line_2 +'\n'
        
        simbolos = ' ,\n./:@'
        text_clean = ''
        for char in text_2:
            if (char.isalnum())| (char in simbolos):
                text_clean += char
        #print(text_clean)
        # Limpiar palabras completamente en mayusculas, es importante
        # para reconocer los nombres
        
        text = text_clean
    return text

if __name__ == '__main__':
    resumes = []
    direc = os.getcwd()
    dir_pdfs = '/parser/resumes_pdf/'
    dir_output = '/parser/Outputs/output_text/'
    
    for root, _, filenames in os.walk(direc + dir_pdfs):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)
    for resume in resumes:
        name = resume.replace(direc+dir_pdfs, '').replace('.pdf', '')
        text = extract_text(resume)
        text_file = open(direc + dir_output + name, "wt",encoding='utf-8')
        n = text_file.write(text)
        text_file.close()