# cv-parser

## Progreso
- [x] Reconocimiento de nombres
- [x] Match de skill de 1 palabras y más de varias palabras.
- [x] Match correo
- [x] Match número telefonico
- [x] Detectar Idiomas
- [x] Exportación formato .JSON
- [x] Detectar licencias y certificaciones
- [x] Match Instituciones de educación chilena, usando nombres y siglas.
- [x] Palabras claves que podrian no estar en su forma raiz. Soy proactivo--> Proactividad
- [x] Detectar el grado académico más alto.
- [x] Recuperar Linkedin
- [ ] Detectar secciones y separar el documento. 
- [ ] Patentes
- [ ] Publicaciones
- [ ] Referencias

## Para ir probando estoy ejecutando el `parser.py`, y este archivo saca  los pdf que se encuentren en la carpeta /resumes
# Usando BETO para hacer resumenes, es solo una idea. Al parecer es pesado de correr
https://colab.research.google.com/drive/11MJSrHQsVDsH1imQ7jGBea9na6m59c7k?usp=sharing

Ejemplo de salida:
```
[{'Celular': ['+56985854563'],
  'Correo': ['email@freecvtemplate.org'],
  'Educacion': ['Universidad de Chile', 'UTEM'],
  'Experiencia Previa': None,
  'Experiencia Previa new': {'Experiencia laboral Técnico electromecanico  '
                             '2015-2019 Entel \uf0a7 Reparación de fibra '
                             'óptica \uf0a7 Organizar y programar brigadas '
                             'Técnico electromecánico '
                             'Siemenes                                                   '
                             '2011-2015 \uf0a7 Encargado de la reparación de '
                             'motores Habilidades \uf0a7 Office 365 \uf0a7 '
                             'Programación en python y c++ \uf0a7 '},
  'Grado': 'Técnico',
  'Hobbies': None,
  'Lenguajes': ['Inglés c1', 'Español nativo', 'Inglés', 'Español'],
  'Licencias y Certificaciones': None,
  'Linkedin': None,
  'Nombre': None,
  'Palabras Claves': {'Creatividad', 'Proactivo', 'Liderazgo', 'Responsible'},
  'Patentes': None,
  'Publicaciones': None,
  'Referencias': None,
  'Skills': ['Python', 'C++', 'Office']},
 {'Celular': ['+56986634232'],
  'Correo': ['erwinpaillacan@gmail.com'],
  'Educacion': ['Universidad de Chile'],
  'Experiencia Previa': None,
  'Experiencia Previa new': {'Experiencia TELECSA Osorno, Región de los Lagos, '
                             'Chile Alumno en Práctica I Enero ’19 – Febrero '
                             '’19 Apoyo en área de transmisión y cooperados. '},
  'Grado': 'Ingeniería',
  'Hobbies': None,
  'Lenguajes': ['Inglés', 'Español'],
  'Licencias y Certificaciones': None,
  'Linkedin': 'https://www.linkedin.com/in/erwinpaillacan',
  'Nombre': 'Erwin Nicolás Paillacán',
  'Palabras Claves': {'Aprender', 'Apasionado'},
  'Patentes': None,
  'Publicaciones': None,
  'Referencias': None,
  'Skills': ['Excel',
             'Git',
             'Octave',
             'Aws',
             'Latex',
             'Pytorch',
             'C',
             'Verilog',
             'Docker',
             'Java',
             'Sql',
             'Python']},
 {'Celular': ['+56995005854'],
  'Correo': ['dasla.pando.f@gmail.com'],
  'Educacion': ['Universidad de Chile', 'UChile'],
  'Experiencia Previa': None,
  'Experiencia Previa new': {'EXPERIENCIA LABORAL 08/2017 – Presente Profesor '
                             'Auxiliar Universidad de Chile Santiago '
                             'Descripción general: Apoyo docente del ramo '
                             'Cálculo Avanzado y Aplicaciones en la Facultad '
                             'de Ciencias Físicas y Matemáticas. '},
  'Grado': 'Ingeniería',
  'Hobbies': None,
  'Lenguajes': ['Español'],
  'Licencias y Certificaciones': None,
  'Linkedin': None,
  'Nombre': 'Dasla Pando Flores',
  'Palabras Claves': {'Adaptabilidad',
                      'Apasionada',
                      'Aprender',
                      'Creatividad',
                      'Disposición',
                      'Liderazgo',
                      'Proactividad',
                      'Resiliencia',
                      'Responsabilidad'},
  'Patentes': None,
  'Publicaciones': None,
  'Referencias': None,
  'Skills': ['Excel',
             'Matlab',
             'Powerpoint',
             'Latex',
             'Java',
             'Arduino',
             'Html',
             'Iot',
             'Python',
             'Word']},
 {'Celular': '+56982655924',
  'Correo': 'andrea.garces.97@hotmail.com',
  'Educacion': ['Universidad de los Lagos'],
  'Experiencia Previa': None,
  'Experiencia Previa new': {'EXPERIENCIA PROFESIONAL \uf0b7 Escuela Rural '
                             'Bahía Mansa, San Juan de La Costa (2019) '
                             'Educadora Diferencial a cargo de realizar apoyos '
                             'en aula común y de recursos en  asignaturas de '
                             'Lenguaje y Matemáticas, cursos 3° y 4° Básico. '},
  'Grado': None,
  'Hobbies': None,
  'Lenguajes': ['Inglés'],
  'Licencias y Certificaciones': None,
  'Linkedin': None,
  'Nombre': 'Andrea Carola Garcés',
  'Palabras Claves': {'Adaptación',
                      'Comprometida',
                      'Paciencia',
                      'Puntual',
                      'Responsable',
                      'Tolerancia'},
  'Patentes': None,
  'Publicaciones': None,
  'Referencias': None,
  'Skills': ['Power point', 'Office', 'Word']}]
```
### Resumen extraido con BETO
```
PERFIL PROFESIONAL Educadora Diferencial, 23 aos de edad, con Especialidad en Dificultades de 
Aprendizaje, titulada de la Universidad de Los Lagos . profesional responsable, comprometida, con
capacidad de adaptación y puntual, con conocimientos en el área administrativa y estrategias did
```

### Resumen extraido con algoritmo simple
```
'EXPERIENCIA PROFESIONAL Escuela Rural Bahía Mansa, San Juan de La Costa (2019)
Educadora Diferencial a cargo de realizar apoyos en aula común y derecursos en  asignaturas de Lenguaje y 
Matemáticas, cursos 3° y 4°Básico.'
```