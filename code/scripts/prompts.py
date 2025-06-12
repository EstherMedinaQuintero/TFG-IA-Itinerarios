import os
import pandas as pd

from langchain_community.document_loaders import DataFrameLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -----------------------------------------------------------------------------
# Módulo: prompts.py
# Descripción: Carga documentos desde Excel, genera un índice FAISS con embeddings
#              de Ollama, configura una cadena RetrievalQA y guarda respuestas en
#              un archivo Markdown para varios prompts formativos.
# Autora: Esther M. Quintero
# -----------------------------------------------------------------------------


def load_documents_from_excel(directory: str) -> list:
    """
    Carga y prepara documentos desde archivos Excel en un directorio.

    Para cada archivo .xlsx en el directorio:
    - Lee el archivo en un DataFrame de pandas.
    - Rellena valores NaN con cadenas vacías.
    - Concatena todas las columnas en una única columna 'content'.
    - Utiliza DataFrameLoader para convertir cada fila en un documento.

    Args:
        directory (str): Ruta al directorio con archivos .xlsx.

    Returns:
        list: Documentos preparados para indexar.
    """
    print("Cargando documentos desde Excel...")
    documents: list = []

    for filename in os.listdir(directory):
        if not filename.endswith(".xlsx"):
            continue

        filepath = os.path.join(directory, filename)
        df = pd.read_excel(filepath)
        df = df.fillna("")

        df["content"] = df.apply(
            lambda row: " | ".join(
                f"{col}: {row[col]}" for col in df.columns
            ),
            axis=1
        )

        loader = DataFrameLoader(df, page_content_column="content")
        documents.extend(loader.load())

    print(f"✔ {len(documents)} documentos cargados.")
    return documents


def build_vectorstore(docs: list) -> FAISS:
    """
    Genera embeddings con OllamaEmbeddings y construye un índice FAISS.

    Args:
        docs (list): Documentos a indexar.

    Returns:
        FAISS: Índice FAISS construido a partir de los documentos.
    """
    print("Generando embeddings y creando el índice FAISS...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("✔ Vectorstore creado correctamente.")
    return vectorstore


def create_retriever_chain(vectorstore: FAISS) -> tuple:
    """
    Configura el modelo y la cadena RetrievalQA con un retriever FAISS.

    Args:
        vectorstore (FAISS): Índice FAISS con embeddings.

    Returns:
        RetrievalQA: Cadena para consulta y recuperación de documentos.
    """
    print("Configurando el modelo y la cadena de recuperación...")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    # llm = OllamaLLM(model="deepseek-r1:14b")
    # llm = OllamaLLM(model="llama2")
    # llm = OllamaLLM(model="gemma2")
    # llm = OllamaLLM(model="phi4")
    # llm = OllamaLLM(model="qwen3:14b")
    # model_name = "gemma3:12b" 
    model_name = "gemma3:1b"
    llm = OllamaLLM(model=model_name)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    # Exponer retriever para usos futuros
    chain.retriever = retriever

    print("✔ Cadena de recuperación lista.")
    return chain, model_name


def generate_markdown_responses(chain: RetrievalQA, model_name: str) -> None:
    """
    Ejecuta una serie de peticiones formativas y guarda las respuestas en Markdown.

    Args:
        chain (RetrievalQA): Cadena configurada para consultas.
    """
    peticiones = [
        # Itinerario 3º Primaria - Matemáticas
        (
            "Genera un itinerario de 1-2 sesiones (hasta 2 horas) dirigido a alumnos"
            " de 3º de Primaria en Matemáticas. Usa las actividades de la base de"
            " conocimiento. Formato: lista numerada, duración total y justificación"
            " breve (≤200 palabras). Añade enlaces (columna 'URL')."
        ),
        # Itinerario 2º ESO - Lengua y Literatura
        (
            "Genera un itinerario de 1-2 sesiones (hasta 2 horas) para estudiantes"
            " de 2º de ESO en Lengua y Literatura. Usa actividades de la base de"
            " conocimiento. Lista, duración y razón breve (≤150 palabras). Añade"
            " enlaces (URL)."
        ),
        # Itinerario 5º Primaria - Ciencias Naturales
        (
            "Genera un itinerario de 1-2 semanas (2-4 sesiones) para 5º de Primaria"
            " en Ciencias Naturales (STEM). Explica cada actividad (≤200 palabras),"
            " indica sesiones totales, recursos y adaptación. Lista numerada con enlaces."
        ),
        # Itinerario 3º ESO - Geografía e Historia
        (
            "Genera un itinerario de 1-2 semanas (2-4 sesiones) para 3º de ESO en"
            " Geografía e Historia. Justifica cada recurso (≤150 palabras). Lista"
            " enumerada con duración de sesiones y enlaces."
        ),
        # Itinerario 6º Primaria - Robótica
        (
            "Crea un itinerario de un mes (5-15 sesiones) para 6º de Primaria en"
            " Robótica (STEM). Justificación breve (≤250 palabras), tiempo estimado"
            " por sesión y encadenamiento de actividades. Lista con enlaces."
        ),
        # Itinerario 4º ESO - Filosofía
        (
            "Diseña un itinerario de un mes (8-10 sesiones) para 4º de ESO en"
            " Filosofía. Describe cada sesión, duración y razón de selección. Total"
            " ≤300 palabras. Lista enumerada con enlaces."
        ),
    ]

    print("\nGenerando archivo Markdown con las respuestas del modelo...\n")
    # respuestas_(nombre modelo usado).md
    output_file = f"respuestas_{model_name}.md"

    with open(output_file, "w", encoding="utf-8") as md_file:
        for idx, peticion in enumerate(peticiones, start=1):
            respuesta = chain.invoke(peticion)

            md_file.write(f"## Pregunta {idx}\n")
            md_file.write(f"**Pedir:** {peticion}\n\n")
            md_file.write("**Respuesta del modelo:**\n\n")
            md_file.write(f"{respuesta}\n\n")
            md_file.write("---\n\n")

    print(f"✔ Archivo '{output_file}' generado con éxito.")


if __name__ == "__main__":
    # Punto de entrada del script
    print("Inicializando el asistente...\n")

    documents = load_documents_from_excel("../../data")
    vector_store = build_vectorstore(documents)
    qa_chain, model_name = create_retriever_chain(vector_store)
    generate_markdown_responses(qa_chain, model_name)
