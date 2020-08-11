import pandas as pd
import os
import sys

grados_educativos = ["técnico", "licenciatura", "enseñanza media", "enseñanza básica", "magister",
                     "msc", "doctorado", "phd", "postdoc", "tecnico profesional"]
## quizas ordenarlos  y buscar el más alto en el curriculum

sections = {"resumen": ["Resumen"],
            "skills": ["Habilidades", "Conocimientos técnicos"],
            "experiencia": ["Experiencia previa", "Antecedentes Laborales"],
            "educacion": ["Educación", "Formación", "Antecedentes de formación"]}

educacion = ["Universidad de chile", "UChile", "UC", "PUC", "Pontificia Universidad Católica de Chile"]

skills = list(pd.read_csv(os.path.join(sys.path[0], "skills.csv")))

print(skills)