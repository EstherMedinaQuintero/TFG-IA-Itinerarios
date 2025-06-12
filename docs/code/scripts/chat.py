import os
import time
from typing import List, Any

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -----------------------------------------------------------------------------
# Módulo: chat.py
# Descripción: Asistente educativo conversacional que carga documentos desde
#              archivos Excel, construye un índice FAISS con embeddings de Ollama
#              y permite interacción de texto en la terminal con contexto.
# Autora: Esther M. Quintero
# -----------------------------------------------------------------------------

def load_documents_from_excel(directory: str) -> List[Any]:
    """
    Carga y prepara documentos desde archivos Excel en un directorio.

    Cada archivo .xlsx se lee en un DataFrame de pandas, se rellenan los valores
    faltantes con cadenas vacías, y se crea una columna 'content' que concatena
    todas las columnas como texto. Los documentos se extraen usando DataFrameLoader.

    Args:
        directory (str): Ruta al directorio que contiene archivos .xlsx.

    Returns:
        List[Any]: Lista de objetos documento listos para indexar.
    """
    print("Cargando documentos desde Excel...")
    documents: List[Any] = []

    for filename in os.listdir(directory):
        if not filename.endswith(".xlsx"):
            continue

        filepath = os.path.join(directory, filename)
        df = pd.read_excel(filepath).fillna("")
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

def build_vectorstore(docs: List[Any]) -> FAISS:
    """
    Genera embeddings y construye el índice FAISS.

    Args:
        docs (List[Any]): Documentos a indexar.

    Returns:
        FAISS: Índice FAISS construido con los embeddings.
    """
    print("Generando embeddings y creando el índice FAISS...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("✔ Vectorstore creado correctamente.")
    return vectorstore

def create_retriever_chain(vectorstore: FAISS) -> RetrievalQA:
    """
    Configura la cadena RetrievalQA con un retriever FAISS y un LLM de Ollama.

    Args:
        vectorstore (FAISS): Índice FAISS con embeddings de documentos.

    Returns:
        RetrievalQA: Cadena configurada para consultas y recuperación de contexto.
    """
    print("Configurando el modelo y la cadena de recuperación...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = OllamaLLM(model="gemma3:1b")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    chain.retriever = retriever  # Exponer retriever para acceso directo

    print("✔ Cadena de recuperación lista.")
    return chain

def handle_conversation(chain: RetrievalQA) -> None:
    """
    Inicia un bucle de conversación con el usuario en la terminal.

    Solicita entradas de usuario, muestra contexto recuperado y la respuesta del LLM.
    El bucle finaliza cuando el usuario escribe 'salir' o 'exit'.

    Args:
        chain (RetrievalQA): Cadena de recuperación para generar respuestas.
    """
    context: str = ""
    print("\nBienvenido al asistente educativo. Escribe 'salir' para terminar.\n")

    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ("salir", "exit"):  # type: ignore
            break

        print("Pensando...", end="", flush=True)
        time.sleep(0.5)

        # Mostrar contexto recuperado
        retrieved_docs = chain.retriever.get_relevant_documents(user_input)
        print("\n[Contexto recuperado]:")
        for idx, doc in enumerate(retrieved_docs, start=1):
            snippet = doc.page_content[:300]  # Mostrar primeros 300 caracteres
            print(f"{idx}. {snippet}...\n")

        # Obtener respuesta del modelo
        result = chain.invoke(user_input)
        print(f"Gemma: {result}")
        context += f"Usuario: {user_input}\nIA: {result}\n"

if __name__ == "__main__":
    # Punto de entrada del script
    print("Inicializando el asistente...\n")

    docs = load_documents_from_excel("./data")
    vectorstore = build_vectorstore(docs)
    chain = create_retriever_chain(vectorstore)
    handle_conversation(chain)