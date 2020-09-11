import os
from utils import extraer_texto




if __name__ == '__main__':
    resumes = []
    direc = os.getcwd()
    dir_pdfs = '/resumes_pdf/'
    dir_output = '/Outputs/output_text/'
    
    path = direc + dir_pdfs
  
    path_out = direc+dir_output
    # Se guardan en una lista los path a los cv en pdf que se ubican en dir_pdfs
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)
    #resumes = [file for file in resumes if file.endswith('.pdf')]
    # Cada pdf se transforma a .txt y se guarda en dir_outputs        
    for resume in resumes:
        name = resume.replace(path, '').replace('.pdf', '')
        text = extraer_texto(resume)
        text_file = open(path_out + name, "wt",encoding='utf-8')
        n = text_file.write(text)
        text_file.close()