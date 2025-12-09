# Proyecto_RI_2025b_Alb-n_Ramirez
# Sistema de RI de Reviews de Steam

En este proyecto se encuentran 3 modelos de Recuperación de la Información clásica que devolveran el top de documentos encontrados por cada algoritmo en un entorno de "CMD"

1.  **Jaccard:** Mide qué tan parecidas son las palabras de tu pregunta y las palabras del documento.
2.  **TF-IDF (Term Frequency-Inverse Document Frequency):** Da más importancia a las palabras que aparecen mucho en un documento, pero poco en toda la colección (las palabras raras, pero importantes).
3.  **BM25:** Una versión mejorada y moderna del TF-IDF que se usa en muchos buscadores grandes.

---

## Diseño del Sistema y Librerías Usadas

El proyecto está dividido varios archivos principales, con el fin de facilitar la comprobación de las ejecuciones, cada uno con una responsabilidad clara. Se usó librerías estándar de Python para el procesamiento de lenguaje:

### 1. `pre_procesamiento.py` 

Prepara los textos y construye la estructura de datos principal, en este caso un índice invertido que usará el buscador.

* **Funciones Clave:**
    * `preprocesar`: Limpieza , tokenización, eliminación de *stop words*, y *stemming* (reducir a la raíz).
    * `construir_indice`: **CAMBIO CLAVE:** Esta función es la única responsable de generar el **Índice Invertido** (el "índice alfabético" detallado) 

* **Librerías Clave:** `nltk` (para el procesamiento de lenguaje), `pandas` (para la manipulación de datos).

### 2. `modelo_ri.py` 

**Propósito:** Implementar los modelos de relevancia y, crucialmente, la lógica de optimización.

* **Funciones de Consulta:** `consulta_jaccard`, `consulta_tfidf`, `consulta_bm25`.

### 3. `evaluacion.py` 

**Propósito:** Medir si nuestro buscador funciona bien, comparando sus resultados contra una lista de respuestas correctas (qrels.json)

### 4. `cli.py` 

**Propósito:** Ejecutar el código fuente en un terminal, cargar corpus y hacer llamadas a las funciones dentro de las otras librerías.

---

## Requisitos de Ejecución

Descargar o clonar el codigo fuente y todas sus carpetas y archivos.

Descomprimir el archivo y almacenar todos los documentos como se muestran

Para que el proyecto funcione, se debe tener Python instalado y las siguientes librerías: pandas y nlt

En caso de no contar con las librerías y para mayor seguridad, dentro de una terminar se deberá ejecutar **"pip install pandas nltk"** o dos comandos separados: **"pip install pandas"** y **"pip install nltk"**

Para iniciar la ejecucióm, dentro de un Simbolo del Sistema (CMD) se deberá dirigir hacia el directorio de la carpeta que tiene los archivos, **Por ejemplo: C:/Users/TuUsuario/Downloads/Proyecto_RI_2025b_Alban_Ramirez**, una vez en la carpeta y con el lenguaje y las librerías descagadas, se escribirá **"python cli.py"** y se iniciará el Sistema de Recuperación de la Información.

## Ejecución

Con el Sistema de Recuperación de la Información se realizará de manera automática la carga del corpus, su preprocesamiento y la construcción del indice invertido.

Se presentará las opciones para **"salir"** y **"evaluar"** a demás del espacio para realizar las consultas.

Las consultas son de texto libre, sin embargo el corpus original está en el idioma **Inglés**, a demás la temática principal es **Videojuegos** por lo que los sistemas de RI tendrán un buen desempeño si las consultas se realizan de acuerdo al tema.

En caso de no existir ninguna coincidencia se devolveran los diocumentos que se clasificaron como máscercanos por cada uno de los algoritmos y su Score o métrica será muy bajo o 0.

Una vez que se realize una consulta se mostrará el top 10 de los documentos recuperados con su ID y un fragmento del texto por cada uno de los modelos de RI y se premitirá realizar un nueva consulta.




