import utils
import os
import es_core_news_sm
import json
import pprint
import nltk
import os
import multiprocessing as mp
import timeit


from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

#Utilidad para limpiar la consola
os.system('cls' if os.name == 'nt' else 'clear')


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')



class CvParser:
    def __init__(self, cv):
        self.parsed = {"Contacto": {'Nombre': None, 'Correo': None, "Celular" : None, "Linkedin": None},
                        "Nombre archivo" : None,
                       "Licencias-Certificaciones": None,
                       "Palabras Claves": None,
                       "Educacion":{'Universidades:': None, 'Grado_mas_alto': None},
                       "Lenguajes": None,
                       "Experiencia": None,
                       "Resumen": None,
                       "Skills": None,
                       "Referencias": None}

        self.cv = cv
        self.raw_text = utils.extraer_texto(self.cv)
        self._nlp = es_core_news_sm.load()
        self.nlp = self._nlp(self.raw_text)
        self.parse()

    def parse(self):
        nombre_archivo = self.cv.replace(direc + dir_pdfs, '').replace('.pdf', '')
        nombre = utils.extraer_nombre(self.raw_text, self.nlp)
        correo = utils.extraer_mail(self.raw_text)
        celular = utils.extraer_fono(self.raw_text)
        skills = utils.extraer_skills(self.raw_text, self.nlp)
        educacion = utils.extraer_educacion(self.raw_text, self.nlp)
        grado = utils.extraer_grado(self.raw_text)
        Lenguajes = utils.extraer_idiomas(self.raw_text, self.nlp)
        Linkedin = utils.extraer_linkedin(self.raw_text)
        palabras_claves = utils.busqueda_palabras_claves(self.raw_text)
        licencias = utils.extraer_licencias(self.raw_text, self.nlp)
        experiencia = utils.extraer_experiencia(self.raw_text)
        resumen = utils.extraer_perfil(self.raw_text)
        referencias = utils.extraer_referencias(self.raw_text)

        self.parsed["Contacto"] = {'Nombre': nombre, 'Correo': correo, 'Celular': celular, "Linkedin": Linkedin}
        self.parsed["Nombre archivo"] = nombre_archivo
        self.parsed["Skills"] = skills
        self.parsed["Educacion"] = {'Universidades:': educacion, 'Grado_mas_alto': grado}
        self.parsed['Lenguajes'] = Lenguajes
        self.parsed["Palabras Claves"] = palabras_claves
        self.parsed['Licencias-Certificaciones']=licencias
        self.parsed['Experiencia']=experiencia
        self.parsed['Resumen']= resumen
        self.parsed['Referencias']= referencias

    def get_parsed_resume(self):
        return self.parsed




def resume_result_wrapper(resume):
    parser = CvParser(resume)
    result = parser.get_parsed_resume()
    name = result["Nombre archivo"]
    with open(direc + dir_output + name +'.json', 'w',encoding='utf-8') as json_file:
        json.dump(result, json_file,ensure_ascii=False,indent=4)
   



if __name__ == '__main__':
    
    pool = mp.Pool(mp.cpu_count())
    resumes = []
    direc = os.getcwd()
    dir_pdfs = '/resumes_pdf/'
    dir_output = '/Outputs/output_parser/'

    #Cargar todos los path a los archivos .pdf ubicados en dir_pdfs
    for root, _, filenames in os.walk(direc + dir_pdfs):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)

    print('Procesando '+str(len(resumes)) + ' CVs')
    
    #Crear un objeto para cada CV y rellenar sus atributos.
    results= [pool.apply_async(resume_result_wrapper(cv), args=(cv,)) for cv in resumes]

 
    
  
    print('Finalizado')