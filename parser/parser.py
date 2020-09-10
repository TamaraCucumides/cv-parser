import utils
import os
import es_core_news_sm
import json
import pprint
import nltk
import os
import multiprocessing as mp
#Utilidad para limpiar la consola
os.system('cls' if os.name == 'nt' else 'clear')


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


class CvParser:
    def __init__(self, cv):
        self.parsed = {"Nombre": None,
                       "Correo": None,
                       "Linkedin": None,
                       "Celular": None,
                       "Skills": None,
                       "Grado": None,
                       "Palabras Claves": None,
                       "Educacion": None,
                       "Lenguajes": None}

        self.cv = cv
        self.raw_text = utils.extract_text(self.cv)
        self._nlp = es_core_news_sm.load()
        self.nlp = self._nlp(self.raw_text)
        self.parse()

    def parse(self):
        nombre_archivo = self.cv.replace(direc + dir_pdfs, '').replace('.pdf', '')
        nombre = utils.retrieve_name(self.raw_text, self.nlp)
        correo = utils.retrieve_email(self.raw_text)
        celular = utils.retrieve_phone_number(self.raw_text)
        skills = utils.retrieve_skills( self.nlp)
        educacion = utils.retrieve_education_institution(self.raw_text, self.nlp)
        grado = utils.retrieve_higher_degree(self.raw_text)
        Lenguajes = utils.retrieve_languages(self.raw_text, self.nlp)
        Linkedin = utils.extract_linkedin(self.raw_text)
        palabras_claves = utils.busqueda_palabras_claves(self.raw_text)


        self.parsed["Nombre"] = nombre
        self.parsed["Nombre archivo"] = nombre_archivo
        self.parsed["Correo"] = correo
        self.parsed["Celular"] = celular
        self.parsed["Skills"] = skills
        self.parsed["Educacion"] = educacion
        self.parsed['Grado']= grado
        self.parsed['Lenguajes'] = Lenguajes
        self.parsed["Linkedin"] = Linkedin
        self.parsed["Palabras Claves"] = palabras_claves

    def get_parsed_resume(self):
        return self.parsed

    def _sectionize(self):
        return utils.parse_cv_sections(self.raw_text)



def resume_result_wrapper(resume):
    parser = CvParser(resume)
    result = parser.get_parsed_resume()
    name = result["Nombre archivo"]
    with open(direc + dir_output + name +'.json', 'w',encoding='utf-8') as json_file:
        json.dump(result, json_file,ensure_ascii=False,indent=4)
   



if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    #print('Usando ' + str(mp.cpu_count()) + ' cores')
    resumes = []
    data = []
    direc = os.getcwd()
    dir_pdfs = '/parser/resumes_pdf/'
    dir_output = '/parser/Outputs/output_parser/'

    #Cargar todos los path a los archivos .pdf ubicados en dir_pdfs
    for root, _, filenames in os.walk(direc + dir_pdfs):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)

    #Crear un objeto para cada CV y rellenar sus atributos.
    #results = [resume_result_wrapper(x) for x in resumes]
    results= [pool.apply_async(resume_result_wrapper(cv), args=(cv,)) for cv in resumes]

    # Exportar toda la informacion extraida a un arhivo .json para cada cv
    
        
    print('Finalizado. Se han procesado '+str(len(results)) + ' CVs')