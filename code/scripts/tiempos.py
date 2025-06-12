import os
import time
import pandas as pd

from langchain_community.document_loaders import DataFrameLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# -----------------------------------------------------------------------------
# Módulo: tiempos.py
# Descripción: Carga documentos desde archivos Excel, genera un vectorstore FAISS
#              usando embeddings de Ollama y ejecuta consultas RetrievalQA
#              para varios modelos, midiendo tiempos de respuesta.
# Autora: Esther M. Quintero
# -----------------------------------------------------------------------------

def load_documents_from_excel(directory: str) -> list:
    """
    Carga y prepara documentos desde archivos Excel en un directorio.

    Para cada archivo .xlsx en el directorio:
    - Lee el archivo en un DataFrame.
    - Rellena valores vacíos con cadenas vacías.
    - Concatena todas las columnas en una única columna 'content'.
    - Utiliza DataFrameLoader para extraer documentos.

    Args:
        directory (str): Ruta al directorio que contiene los archivos .xlsx.

    Returns:
        list: Lista de objetos Documento preparados para indexar.
    """
    print("Cargando documentos desde Excel...")
    documents = []

    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(directory, filename)
            df = pd.read_excel(filepath)
            df = df.fillna("")

            # Construir columna 'content' concatenando nombre de columna y valor
            df["content"] = df.apply(
                lambda row: " | ".join(
                    f"{col}: {row[col]}" for col in df.columns
                ),
                axis=1
            )

            # Cargar documentos desde DataFrame
            loader = DataFrameLoader(df, page_content_column="content")
            documents.extend(loader.load())

    print(f"✔ {len(documents)} documentos cargados.")
    return documents


def build_vectorstore(docs: list) -> FAISS:
    """
    Genera embeddings usando OllamaEmbeddings y crea un vectorstore FAISS.

    Args:
        docs (list): Lista de documentos a indexar.

    Returns:
        FAISS: Índice FAISS construido a partir de los embeddings.
    """
    print("Generando embeddings y creando el índice FAISS...")

    # Inicializar embeddings Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Construir índice FAISS a partir de documentos
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("✔ Vectorstore creado correctamente.")
    return vectorstore


def main() -> None:
    """
    Función principal que coordina la carga de datos, construcción de índice,
    ejecución de prompts contra múltiples modelos y guardado de resultados.

    Pasos:
    1) Cargar documentos desde Excel.
    2) Generar vectorstore FAISS y obtener un retriever.
    3) Definir modelos y prompts a ejecutar.
    4) Iterar cada modelo y prompt, medir el tiempo de ejecución.
    5) Guardar tiempos y respuestas en CSV y Markdown.
    """
    # 1) Cargar datos y construir el índice
    docs = load_documents_from_excel("./data")
    vectorstore = build_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 2) Definir lista de modelos y prompts
    modelos = [
        "gemma3:1b",
        "llama3.2",
        "phi3",
        "moondream",
        "deepseek-r1:1.5b"
    ]

    peticiones = [
        "Genera un itinerario de 1-2 sesiones (hasta 2 horas) dirigido a alumnos de 3º de Primaria en la asignatura de Matemáticas. Tienes que usar las actividades de la base de conocimiento que te he pasado. Redacta la respuesta en formato de lista numerada, indicando la duración total y justificando brevemente cada recurso (máximo 200 palabras). Asegúrate de explicar por qué cada actividad se ajusta al nivel de 3º de Primaria. Escribe la respuesta en Español. Tienes que añadir el enlace de las actividades que selecciones. Los enlaces están en la columna “URL”.",
        "Genera un itinerario de 1-2 sesiones (hasta 2 horas) para estudiantes de 2º de ESO en la asignatura de Lengua y Literatura. Tienes que usar las actividades de la base de conocimiento que te he pasado. Presenta la respuesta en formato de lista, describiendo la duración total y explicando brevemente la razón de incluir cada recurso (máximo 150 palabras). Asegúrate de que se ajuste a 2º de ESO y no supere 2 sesiones. Escribe la respuesta en Español. Tienes que añadir el enlace de las actividades que selecciones. Los enlaces están en la columna “URL”.",
        "Genera un itinerario de 1-2 semanas (entre 2 y 4 sesiones) para 5º de Primaria, centrado en Ciencias Naturales (rama STEM). Tienes que usar las actividades de la base de conocimiento que te he pasado. Debes explicar cada actividad con un máximo de 200 palabras. Indica cuántas sesiones totales se necesitan, qué recursos se usan y por qué se adaptan a 5º de Primaria. Responde en formato numerado. Escribe la respuesta en Español. Tienes que añadir el enlace de las actividades que selecciones. Los enlaces están en la columna “URL”.",
        "Genera un itinerario de 1-2 semanas (2-4 sesiones) para 3º de ESO en la asignatura de Geografía e Historia. Tienes que usar las actividades de la base de conocimiento que te he pasado. Justifica el uso de cada recurso en menos de 150 palabras y no incluyas referencias ficticias. Escribe tu respuesta como una lista enumerada que especifique la duración aproximada de cada sesión y cómo se integran los recursos seleccionados. Escribe la respuesta en Español. Tienes que añadir el enlace de las actividades que selecciones. Los enlaces están en la columna “URL”.",
        "Crea un itinerario formativo de un mes de duración (5-15 sesiones) para estudiantes de 6º de Primaria en la asignatura de Robótica (STEM). Tienes que usar las actividades de la base de conocimiento que te he pasado. Presenta la justificación de cada uno en menos de 250 palabras. Indica el tiempo estimado para cada sesión y explica cómo se encadenan las actividades. Utiliza lista enumerada. Escribe la respuesta en Español. Tienes que añadir el enlace de las actividades que selecciones. Los enlaces están en la columna “URL”.",
        "Diseña un itinerario de un mes de duración (aprox. 8-10 sesiones) para 4º de ESO en la asignatura de Filosofía. Tienes que usar las actividades de la base de conocimiento que te he pasado. Describe en formato de lista cada sesión, su duración estimada y la razón de escoger esos recursos. Asegúrate de no superar 300 palabras en total. Explica brevemente por qué consideras que estas actividades se ajustan al temario de Filosofía para 4º de ESO. Escribe la respuesta en Español. Tienes que añadir el enlace de las actividades que selecciones. Los enlaces están en la columna “URL”."
    ]

    # 3) Iterar modelos y peticiones, midiendo tiempos
    resultados = []
    for modelo in modelos:
        llm = OllamaLLM(model=modelo)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )
        print(f"\nEjecutando prompts con el modelo '{modelo}'...")

        for idx, peticion in enumerate(peticiones, start=1):
            start = time.perf_counter()
            respuesta = chain.invoke(" ".join(peticion))
            elapsed = time.perf_counter() - start

            resultados.append({
                "modelo": modelo,
                "prompt_id": idx,
                "tiempo_segundos": elapsed,
                "respuesta": respuesta
            })

            print(f"Modelo {modelo} - Prompt {idx} completado en {elapsed:.2f}s")

    # 4) Guardar resultados en CSV y Markdown
    df = pd.DataFrame(resultados)
    df.to_csv("resultados_tiempo.csv", index=False)

    with open("resultados_tiempo.md", "w", encoding="utf-8") as md:
        for _, row in df.iterrows():
            md.write(f"**Modelo:** {row['modelo']}  \n")
            md.write(f"**Prompt {row['prompt_id']}:** tiempo {row['tiempo_segundos']:.2f}s  \n")
            md.write("---\n")

    print("✔ Resultados guardados en 'resultados_tiempo.csv' y 'resultados_tiempo.md'.")


if __name__ == "__main__":
    main()
