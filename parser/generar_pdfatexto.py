import os
from utils import extraer_texto



if __name__ == '__main__':
    resumes = []
    direc = os.getcwd()
    dir_pdfs = '/parser/resumes_pdf/'
    dir_output = '/parser/Outputs/output_text/'
    

    # Se guardan en una lista los path a los cv en pdf que se ubican en dir_pdfs
    for root, _, filenames in os.walk(direc + dir_pdfs):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)
    #resumes = [file for file in resumes if file.endswith('.pdf')]
    # Cada pdf se transforma a .txt y se guarda en dir_outputs        
    for resume in resumes:
        name = resume.replace(direc+dir_pdfs, '').replace('.pdf', '')
        text = extraer_texto(resume)
        text_file = open(direc + dir_output + name, "wt",encoding='utf-8')
        n = text_file.write(text)
        text_file.close()