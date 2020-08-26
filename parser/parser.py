import utils
import os
import en_core_web_sm

import pprint

class CvParser:
    def __init__(self, cv):
        nlp = en_core_web_sm.load()
        self.parsed = {"Nombre": None,
                       "Correo": None,
                       "Celular": None,
                       "Skills": None,
                       "Experiencia Previa": None,
                       "Educacion": None}
        self.cv = cv
        self.raw_text = utils.extract_text(self.cv)
        self.text = ' '.join(self.raw_text.split())
        self.nlp = nlp(self.text)
        self.parse()
        self._sections = dict()

    def parse(self):
        nombre = utils.retrieve_name(self.raw_text)
        correo = utils.retrieve_email(self.raw_text)
        celular = utils.retrieve_phone_number(self.raw_text)
        skills = utils.retrieve_skills(self.raw_text)
        experiencia = utils.retrieve_past_experience(self.raw_text)
        educacion = utils.retrieve_education_institution(self.raw_text)

        self.parsed["Nombre"] = nombre
        self.parsed["Correo"] = correo
        self.parsed["Celular"] = celular
        self.parsed["Skills"] = skills
        self.parsed["Experiencia Previa"] = experiencia
        self.parsed["Educacion"] = educacion

    def get_parsed_resume(self):
        return self.parsed

    def _sectionize(self):
        return utils.parse_cv_sections(self.raw_text)

#resume = '/home/erwin/cv-parser/example_1.pdf'
#parser = CvParser(resume)
#parser.parse()

def resume_result_wrapper(resume):
        parser = CvParser(resume)
        return parser.get_parsed_resume()



if __name__ == '__main__':
    resumes = []
    data = []
    for root, directories, filenames in os.walk('resumes'):
        for filename in filenames:
            file = os.path.join(root, filename)
            resumes.append(file)
    #print(resumes)

    results = [resume_result_wrapper(x) for x in resumes]
    print(results)

    #results = [p.get_parse for p in results]

    #pprint.pprint(results)








