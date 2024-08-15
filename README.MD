# Chatear con el PDF de la cátedra

## Funcionamiento

ETL de construcción
- Primero se lee el archivo de la cátedra y se divide en páginas
- Luego se indexan esas páginas utilizando BM25 y Chroma con nomic embeddings
    - todo esto sucede en el constructor en el método `__init__`

ETL de Prueba
- Hay 3 operaciones para generar los resultados:
    - rag_with_ensemble
    - rag_with_bm25
    - rag_with_chroma
- Todas estas operaciones siguen los mismos pasos: 
    - recibir una pregunta
    - consultar al índice correspondiente los valores más cercanos
    - generar los resultados de la pregunta

### Problemas
- No se está evaluando el contexto (hay modelos que no tienen un buen manejo contextual)
- estamos asumiendo que nomic es un buen embedding
- no estamos procesando más eficientemente el pdf, podríamos dividirlo en chunks en lugar de en páginas
- no evaluamos otras métricas de similitud (cosine, mmr, etc)
- No tenemos ningún "guardrail" para corroborar que la pregunta no está interfiriendo en el prompt (prompt injection)
- No estamos realizando una evaluación de la respuesta (métrica), solo generamos los resultados de la pregunta.

## Ejecutar Local
Instalar
- python: https://www.python.org/downloads/ o https://github.com/pyenv/pyenv
- ollama: https://ollama.com/download 

Luego ejecutar: 
```
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Descargar los modelos que vayan a utilizar, esto suele tardar pero al menos descargar llama3.1:8b u otro más chico.
```
ollama pull ollama3.1:8b
```

Para correr el proyecto:
```
python script.py        # esto va a ejecutar todo lo necesario para crear la db vectorial
streamlit run view.py   # esto va a ejecutar un frontend en streamlit para hacerle preguntas al pdf con distintos modelos
```

## Ejecutar Colab

Ir al archivo [notebook](./TABI_Chatear_con_el_Libro_de_Cátedra_(2024).ipynb) y presionar el botón "open in colab". 
Luego ejecutar todas las celdas. Elegir un entorno de ejecución con GPU/TPU para que sea un poco más rápido.