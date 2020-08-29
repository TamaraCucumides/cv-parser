import pandas as pd
import os
import sys

grados_educativos = ["técnico", "licenciatura", "enseñanza media", "enseñanza básica", "magister", 'Educación Media', 'Educación Básica'
                     "msc", "doctorado", "phd", "postdoc", "tecnico profesional", "Ingeniería", "Pedagogía"]
## quizas ordenarlos  y buscar el más alto en el curriculum

grados_educativos_orden = ['enseñanza básica', 'educación básica', 'enseñanza media', 'educacion media', 'técnico', 'tecnico profesional'
                            'licenciatura', 'ingeniería', 'msc', 'magister', 'doctor', 'doctorado', 'phd', 'postdoc']

sections = {"resumen": ["Resumen"],
            "skills": ["Habilidades", "Conocimientos técnicos", "Conocimientos"],
            "experiencia": ["Experiencia previa", "Antecedentes Laborales"],
            "educacion": ["Educación", "Formación", "Antecedentes de formación"]}

educacion = ["Universidad de Chile", "UChile","Pontificia Universidad Católica de Chile",
                "Universidad de los Lagos", "Universidad de Santiago de Chile", "Universidad Tecnológica Metropolitana", "Universidad Diego Portales"
                , 'Universidad del Desarrollo', 'Universidad Austral de Chile', 'Inacap']

educacionSiglas = ['UC', 'UTEM', 'UAI', 'PUC', 'UCH', 'UDD', 'UAI', 'ULA', 'USACH', 'UACH', 'DUOC']

idiomas = ['inglés', 'alemán', 'portugués', 'francés', 'chino', 'mandarin', 'español']
idiomas_nivel = [' a1', ' a2', ' b1', ' b2', ' c1', ' c2', ' básico', ' intermedio', ' avanzado', ' nativo', ' competencia básica' 
                    ' competencia profesional']

skills = list(pd.read_csv(os.path.join(sys.path[0], "skills.csv")))

palabras_claves = ['proactividad', 'liderar', 'liderazgo',  'puntual'
         ,  'paciencia', 'resiliencia', 'resilisente', 'tolerancia', 'comprometido', 'compromiso', 'empatia', 'responsabilidad',
         'negociación', 'dialogo', 'escuchar', 'comunicacion','analizar', 'autodidacta', 'autoaprendizaje', 'adaptación'
         , 'creativo' , 'ética', 'disciplina', 'confianza' , 'motivación', 'colaboración', 'aprender', 'pasion', 'apasionado']