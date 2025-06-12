import os
import re
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import DataFrameLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -----------------------------------------------------------------------------
# Módulo: interfaz.py
# Descripción: Streamlit app para construir itinerarios formativos
#              basados en selecciones de usuario y RetrievalQA
# Autora: Esther M. Quintero
# -----------------------------------------------------------------------------

# Constantes de configuración
DISTRIBUTIONS: dict = {
    "Resource type": [
        "Actividad desenchufada", "Programación visual", "Actividad enchufada",
        "Curso", "Vídeo", "Dispositivos físicos", "Programación textual",
    ],
    "Values": [
        "Pensamiento crítico", "Fomento de la creatividad y del espíritu científico",
        "Trabajo en equipo", "Buen uso de las TIC", "Convivencia y Educación Cívica",
        "Atención a la diversidad", "Educación Ambiental y desarrollo sostenible",
        "Interculturalidad", "Igualdad de Género", "Educación para la Salud",
        "Fomento de la creatividad",
    ],
    "Key competences": [
        "Aprender a Aprender", "Competencia Matemática y en Ciencia y Tecnología",
        "Competencia Digital", "Sociales y Cívicas", "Comunicación Lingüística",
        "Plurilingüe",
    ],
    "Knowledge areas": [
        "Informática y Robótica", "Matemáticas", "Ética", "Lengua y Literatura",
        "Arte", "Biología y Geología", "Filosofía", "Geografía e Historia",
        "Música", "Deporte", "Física", "Tecnología",
        "Ciencias (Biología y Geología)", "Física y Química",
        "Física (dentro de Biología y Geología)",
    ],
    "Cc concepts": [
        "Algoritmia y Programación", "Análisis de Datos",
        "Impacto de la Computación", "Sistemas Informáticos",
        "Redes e Internet",
    ],
    "Ct skills": [
        "Razonamiento Lógico", "Descomposición", "Evaluación",
        "Pensamiento Algorítmico y Programación", "Abstracción", "Patrones",
    ],
    "Duration": [
        "Actividad Rápida (una sola clase)", "Sesión (hasta 2 horas)",
        "Semana (2-4 sesiones)", "Mes (5-15 sesiones)",
        "2 Meses (15-30 sesiones)", "Más de 3 Meses (Más de 30 sesiones)",
    ],
    "School years": [
        "1º ESO", "Tercer Ciclo EP", "2º ESO", "3º ESO", "Segundo Ciclo EP",
        "4º ESO", "1º Bachillerato", "2º Bachillerato", "Primer Ciclo EP",
        "Educación Infantil",
    ],
    "Languages": ["Inglés", "Español"],
}

MODEL_OPTIONS: list = [
    "gemma3:1b",
    "llama3.2",
    "phi3",
    "moondream",
    "deepseek-r1:1.5b",
]

@st.cache_data(ttl=3600)
def load_documents_from_excel(directory: str) -> list:
    """
    Carga documentos de archivos Excel y genera una lista de documentos.

    Args:
        directory (str): Ruta al directorio con archivos .xlsx.

    Returns:
        list: Documentos extraídos con DataFrameLoader.
    """
    documents: list = []

    for filename in os.listdir(directory):
        if not filename.lower().endswith(".xlsx"):
            continue

        filepath = os.path.join(directory, filename)
        dataframe = pd.read_excel(filepath).fillna("")
        dataframe["content"] = dataframe.apply(
            lambda row: " | ".join(
                f"{col}: {row[col]}" for col in dataframe.columns
            ),
            axis=1
        )
        loader = DataFrameLoader(
            dataframe, page_content_column="content"
        )
        documents.extend(loader.load())

    return documents

@st.cache_resource
def build_chain(_documents: list, model_name: str) -> RetrievalQA:
    """
    Construye la cadena RetrievalQA con FAISS y embeddings de Ollama.

    Args:
        documents (list): Documentos a indexar.
        model_name (str): Nombre del modelo Ollama.

    Returns:
        RetrievalQA: Objeto configurado para consultas.
    """
    documents = _documents
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model=model_name)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
    )

def build_prompt(selections: dict) -> str:
    """
    Genera el prompt en Español basado en las selecciones del usuario.

    Args:
        selections (dict): Mapeo de categoría a opciones seleccionadas.

    Returns:
        str: Prompt formateado para enviar al modelo.
    """
    lines: list = ["Genera un itinerario formativo que cumpla estos criterios:"]

    for category, opts in selections.items():
        lines.append(f"- **{category}**: {', '.join(opts)}")

    lines.append(
        "Sintetiza el programa en lista numerada, indicando duración y justificando "
        "cada recurso. Incluye los enlaces de las actividades (columna 'URL') "
        "y redacta en Español."
    )

    return "\n".join(lines)

# Interfaz Streamlit
st.title("🧩 Constructor de Itinerarios Formativos")

st.sidebar.header("Configuración del LLM")
selected_model: str = st.sidebar.selectbox(
    "Selecciona el modelo de Ollama",
    MODEL_OPTIONS,
    index=MODEL_OPTIONS.index("deepseek-r1:1.5b"),
)

st.sidebar.header("Selecciona características")
user_selections: dict = {}
for category, options in DISTRIBUTIONS.items():
    chosen = st.sidebar.multiselect(label=category, options=options, default=[])
    if chosen:
        user_selections[category] = chosen

if st.sidebar.button("Generar Itinerario"):
    if not user_selections:
        st.warning("🚨 Selecciona al menos una característica en la barra lateral.")
    else:
        prompt_text = build_prompt(user_selections)
        st.subheader("🔍 Prompt generado")
        st.code(prompt_text, language="markdown")

        with st.spinner(f"Invocando al modelo {selected_model}…"):
            docs = load_documents_from_excel("./data")
            qa_chain = build_chain(docs, selected_model)
            raw_response = qa_chain.invoke(prompt_text)

        # Extraer texto y limpiar etiquetas internas
        if isinstance(raw_response, dict) and "result" in raw_response:
            content = raw_response["result"]
        else:
            content = raw_response

        cleaned = re.sub(
            r"<think>.*?</think>", "", content, flags=re.S
        ).strip()

        st.subheader("✍️ Respuesta del Modelo")
        st.markdown(cleaned)
