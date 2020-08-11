from . import utils


class CvParser:
    def __init__(self, cv):
        self.parsed = {"Nombre": None,
                       "Correo": None,
                       "Celular": None,
                       "Skills": None,
                       "Experiencia Previa": None,
                       "Educacion": None}
        self.raw_text = cv

        self.parse()
        pass

    def parse(self):
        nombre = None
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












