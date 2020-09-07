import utils
import os
import es_core_news_sm
import json
import pprint
import csv
import nltk
import os
os.system('cls' if os.name == 'nt' else 'clear')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


class CvParser:
    def __init__(self, cv):
        #nlp = en_core_web_sm.load()
        self.parsed = {"Nombre": None,
                       "Correo": None,
                       "Linkedin": None,
                       "Celular": None,
                       #"Resumen": None,
                       "Skills": None,
                       "Grado": None,
                       "Palabras Claves": None,
                      # "Experiencia Previa": None,
                       #"Experiencia Previa new": None,
                       "Educacion": None,
                       #"Licencias y Certificaciones": None,
                       "Lenguajes": None}
                       #"Hobbies": None,
                       #"Referencias": None,
                       #"Patentes": None,
                       #"Publicaciones": None}
        self.cv = cv
        self.raw_text = utils.extract_text(self.cv)
        #self.text = ' '.join(self.raw_text.split())
        self._nlp = es_core_news_sm.load()
        self.nlp = self._nlp(self.raw_text)
        self.parse()
        self._sections = dict()

    def parse(self):
        nombre_archivo = self.cv.replace("resumes/", '').replace('.pdf', '')
        nombre = utils.retrieve_name(self.raw_text, self.nlp)
        correo = utils.retrieve_email(self.raw_text)
        celular = utils.retrieve_phone_number(self.raw_text)
        skills = utils.retrieve_skills( self.nlp)
        #experiencia = utils.retrieve_past_experience(self.raw_text)
        educacion = utils.retrieve_education_institution(self.raw_text, self.nlp)
        grado = utils.retrieve_higher_degree(self.raw_text)
        #resumen = utils.summarize_cv(self.raw_text, self.nlp)
        Lenguajes = utils.retrieve_languages(self.raw_text, self.nlp)
        Linkedin = utils.extract_linkedin(self.raw_text)
        palabras_claves = utils.busqueda_palabras_claves(self.raw_text)
        #experiencia_2 = utils.retrieve_experience_2(self.raw_text)


        self.parsed["Nombre"] = nombre
        self.parsed["Nombre archivo"] = nombre_archivo
        self.parsed["Correo"] = correo
        self.parsed["Celular"] = celular
        self.parsed["Skills"] = skills
        #self.parsed["Experiencia Previa"] = experiencia
        self.parsed["Educacion"] = educacion
        self.parsed['Grado']= grado
        #self.parsed['Resumen']= resumen
        self.parsed['Lenguajes'] = Lenguajes
        self.parsed["Linkedin"] = Linkedin
        self.parsed["Palabras Claves"] = palabras_claves
        #self.parsed["Experiencia Previa new"] = experiencia_2

    def get_parsed_resume(self):
        return self.parsed

    def _sectionize(self):
        return utils.parse_cv_sections(self.raw_text)



def resume_result_wrapper(resume):
        parser = CvParser(resume)
        return parser.get_parsed_resume()



if __name__ == '__main__':
    resumes = []
    data = []
    for root, _, filenames in os.walk(os.getcwd()+'/resumes'):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)

    
    results = [resume_result_wrapper(x) for x in resumes]

    for result in results:
        name = result["Nombre archivo"].replace(os.getcwd(),'')
        with open(os.getcwd()+'/output_parser'+name+'.json', 'w',encoding='utf-8') as json_file:
            json.dump(result, json_file,ensure_ascii=False)
    print('Finalizado. Se han procesado '+str(len(results)) + ' CVs')







