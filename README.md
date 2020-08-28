# cv-parser

## Para ir probando estoy ejecutando el `parser.py`, y este archivo saca  los pdf que se encuentren en la carpeta /resumes
# Usando BETO para hacer resumenes, es solo una idea. Al parecer es pesado de correr
https://colab.research.google.com/drive/11MJSrHQsVDsH1imQ7jGBea9na6m59c7k?usp=sharing

Ejemplo de salida:
```
[{'Celular': ['+56986634232'],
  'Correo': ['erwinpaillacan@gmail.com'],
  'Educacion': ['Universidad de chile'],
  'Experiencia Previa': [],
  'Grado': ['ingeniería'],
  'Nombre': 'Erwin Nicolás Paillacán',
  'Resumen': 'Antecedentes PersonalesFecha de Nacimiento:7 de Enero del '
             '1997Edad:23 añosNacionalidad:ChilenoCédula de '
             'Identidad:19.270.676-9EducaciónUniversidad de ChileSantiago, '
             'Región Metropolitana, ChileLicenciatura en Ciencias de la '
             'Ingeniería, Mención Eléctrica2015 – 2018Minor '
             'Computación–Ingeniería Civil Eléctrica, Telecomunicaciones e '
             'Inteligencia Computacional2015 – 2021Czech Technical University '
             'in PraguePrague, Czech RepublicExchange Semester '
             '2020-1–Conocimientos & HabilidadesUso de software: Excel, LATEX, '
             'Octave, HFSS, NI AWR Design Environment, Ltspice, OMNeT++, GIT.',
  'Skills': ['Latex',
             'Octave',
             'Pytorch',
             'C',
             'Sql',
             'Git',
             'Verilog',
             'Java',
             'Docker',
             'Excel',
             'Aws',
             'Python']},
 {'Celular': ['+56986634232'],
  'Correo': ['erwinpaillacan@gmail.com'],
  'Educacion': ['Universidad de chile'],
  'Experiencia Previa': ['TELECSA Osorno'],
  'Grado': ['ingeniería'],
  'Nombre': 'Erwin Paillacán Huaitro',
  'Resumen': '180 HorasEducaciónUniversidad de ChileSantiago, Región '
             'Metropolitana, ChileLicenciatura en Ciencias de la Ingeniería, '
             'Mención Eléctrica2015 – 2018Ingeniería Civil Eléctrica2015 – '
             '2020Conocimientos & HabilidadesUso de software: Excel, LATEX, '
             'Octave, HFSS, NI AWR Design Environment, Ltspice, OMNeT++, GIT.',
  'Skills': ['Latex',
             'Octave',
             'Pytorch',
             'C',
             'Sql',
             'Git',
             'Verilog',
             'Java',
             'Excel',
             'Python']},
 {'Celular': ['+56986634232'],
  'Correo': ['erwinpaillacan@gmail.com'],
  'Educacion': ['Universidad de chile'],
  'Experiencia Previa': [],
  'Grado': ['ingeniería'],
  'Nombre': 'Erwin Nicolás Paillacán',
  'Resumen': 'Fecha de ti-tulación estimada 2021-1Antecedentes PersonalesFecha '
             'de Nacimiento:7 de Enero del 1997Edad:23 '
             'añosNacionalidad:ChilenoCédula de '
             'Identidad:19.270.676-9EducaciónUniversidad de ChileSantiago, '
             'Región Metropolitana, ChileLicenciatura en Ciencias de la '
             'Ingeniería, Mención Eléctrica2015 – 2018Minor '
             'Computación–Ingeniería Civil Eléctrica, Telecomunicaciones e '
             'Inteligencia Computacional2015 – 2021Czech Technical University '
             'in PraguePrague, Czech RepublicExchange Semester '
             '2020-1–Conocimientos & HabilidadesUso de software: Excel, LATEX, '
             'Octave, HFSS, NI AWR Design Environment, Ltspice, OMNeT++, GIT.',
  'Skills': ['Latex',
             'Octave',
             'Pytorch',
             'C',
             'Sql',
             'Git',
             'Verilog',
             'Java',
             'Docker',
             'Excel',
             'Aws',
             'Python']},
 {'Celular': ['+56982655924', '+56982654303'],
  'Correo': ['andrea.garces.97@hotmail.com', 's.muñoz@ulagos.cl'],
  'Educacion': ['Universidad de los Lagos'],
  'Experiencia Previa': ['PROFESIONAL \uf0b7 Escuela Rural Bahía Mansa'],
  'Grado': [],
  'Nombre': 'Andrea Carola Garcés',
  'Resumen': 'EXPERIENCIAPROFESIONAL\uf0b7Escuela Rural Bahía Mansa, San Juan '
             'de La Costa (2019)Educadora Diferencial a cargo de realizar '
             'apoyos en aula común y derecursos en  asignaturas de Lenguaje y '
             'Matemáticas, cursos 3° y 4°Básico.',
  'Skills': ['Word', 'Power point', 'Office']}]

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