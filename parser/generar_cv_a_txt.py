import os
from utils import extraer_texto




if __name__ == '__main__':
    resumes = []
    direc = os.getcwd()
    dir_files = '/resumes/'
    dir_output = '/Outputs/output_text/'
    
    path = direc + dir_files
  
    path_out = direc + dir_output
    # Se guardan en una lista los path a los cv 
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)
      
    for resume in resumes:
        if resume.endswith('.pdf'):
            name = resume.replace(path, '').replace('.pdf', '')
        elif resume.endswith('.docx'):
            name = resume.replace(path, '').replace('.docx', '')
        else:
            name = resume.replace(path, '').replace('.doc', '')

        text = extraer_texto(resume)
        text_file = open(path_out + name + '.txt', "wt",encoding='utf-8')
        n = text_file.write(text)
        text_file.close()