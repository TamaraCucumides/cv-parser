# CV parser
## Estructura
```
CV parser
├── 
│   ├── Desarollo
│   │   └── test_seccionar.ipynb
|   |   └── test_ranking.ipynb
│   └── parser
│       ├── CSVs
|       |    └── seccionesCV.csv
│       └── Descricpion_cargo
|       |    └── descripcion_cargo.txt
│       └── diccionarios
|       |    └── grados_educativos_orden.txt
|       |    └── idiomas.txt
|       |    └── idiomas_nivel.txt
|       |    └── licencias_certificaciones.txt
|       |    └── palabras claves.txt
|       |    └── skills.txt
|       |    └── stop_words.txt
|       |    └── universidades.txt
|       |    └── universidades_siglas.txt
|       └── embeddings
|       |     └──fasttext-sbwc.3.6.e20.vec
|       └── Outputs
|       |    └── output_parser
|       |    └── output_seccionado
|       |    └── output_text
|       └── resumes_pdf
|       └── cts.py
|       └── generate_text_files.py
|       └── ranking.py
|       └── seccioanr.py
|       └── utils.py
```

La carpeta `/Desarrollo` tiene jupyter notebook que uso para ir probando ideas de forma rápida.

### Dependencias
Para crear un ambiente llamado cv_parser en conda con todo lo necesario.
* `conda env create --name cv_parser -f cv_parser_export.yml`




Probablemente tambien pida esto la primera vez: 
- `conda activate cv_parser`
- `python -m spacy download es_core_news_sm`
- `python -m spacy download es_core_news_md`

### FastText embeddings from SBWC
El archivo `fasttext-sbwc.3.6.e20.vec` se encuentra en el siguiente link (descomprimir):
http://dcc.uchile.cl/~jperez/word-embeddings/fasttext-sbwc.vec.gz




### ¿Cómo usar?

1. Poner todos los CV en la carpeta `/resumes_pdf`.
2. Correr `generate_text_files.py`. La salida (`output_text`) de este script es usada en `seccionar.py`
3. Correr `parser.py`, que usa diccionarios y genera .json con filtros especificos (`output_parser`).
4. Correr `seccionar.py`, que usa la salida del paso 2 y genera .json con los cvs seccionados (`output_seccionado`).
5. Correr `ranking.py` que usa la salida de 4 y `descripcion_cargo.txt`..