# cv-parser

## Para ir probando estoy ejecutando el `parser.py`, y este archivo saca  los pdf que se encuentren en la carpeta /resumes
# Usando BETO para hacer resumenes, es solo una idea. Al parecer es pesado de correr
https://colab.research.google.com/drive/11MJSrHQsVDsH1imQ7jGBea9na6m59c7k?usp=sharing

Ejemplo de salida:
```
[{'Celular': ['+56986634232'],
  'Correo': ['erwinpaillacan@gmail.com'],
  'Educacion': ['Universidad de Chile'],
  'Experiencia Previa': [],
  'Grado': 'Ingeniería',
  'Hobbies': None,
  'Lenguajes': ['Inglés', 'Español'],
  'Licencias y Certificaciones': None,
  'Linkedin': 'https://www.linkedin.com/in/erwinpaillacan',
  'Nombre': 'Erwin Nicolás Paillacán',
  'Palabras Claves': {'Proactividad', 'Responsabilidad', 'Resilisente'},
  'Patentes': None,
  'Publicaciones': None,
  'Referencias': None,
  'Skills': ['Aws',
             'Latex',
             'Java',
             'Docker',
             'Python',
             'Octave',
             'Git',
             'C',
             'Excel',
             'Pytorch',
             'Verilog',
             'Sql']},
 {'Celular': '+56982655924',
  'Correo': 'andrea.garces.97@hotmail.com',
  'Educacion': ['Universidad de los Lagos'],
  'Experiencia Previa': ['PROFESIONAL \uf0b7 Escuela Rural Bahía Mansa'],
  'Grado': None,
  'Hobbies': None,
  'Lenguajes': ['Inglés'],
  'Licencias y Certificaciones': None,
  'Linkedin': '',
  'Nombre': 'Andrea Carola Garcés',
  'Palabras Claves': {'Comprometida',
                      'Paciencia',
                      'Puntual',
                      'Responsable',
                      'Tolerancia'},
  'Patentes': None,
  'Publicaciones': None,
  'Referencias': None,
  'Skills': ['Power point', 'Word', 'Office']}]
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