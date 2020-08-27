import pandas as pd
import os
import sys

grados_educativos = ["técnico", "licenciatura", "enseñanza media", "enseñanza básica", "magister", 'Educación Media', 'Educación Básica'
                     "msc", "doctorado", "phd", "postdoc", "tecnico profesional", "Ingeniería", "Pedagogía"]
## quizas ordenarlos  y buscar el más alto en el curriculum

sections = {"resumen": ["Resumen"],
            "skills": ["Habilidades", "Conocimientos técnicos", "Conocimientos"],
            "experiencia": ["Experiencia previa", "Antecedentes Laborales"],
            "educacion": ["Educación", "Formación", "Antecedentes de formación"]}

educacion = ["Universidad de chile", "UChile", "UCV", "PUC", "Pontificia Universidad Católica de Chile",
                "Universidad de los Lagos", "UTEM"]

skills = list(pd.read_csv(os.path.join(sys.path[0], "skills.csv")))
