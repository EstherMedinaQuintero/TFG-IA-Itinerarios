import os
import re
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import DataFrameLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -----------------------------------------------------------------------------
# Módulo: interfaz_optimizada.py
# Descripción: Streamlit app optimizada para construir itinerarios formativos
#              con tiempos de respuesta reducidos usando caching y estructuras
# Autora: Esther M. Quintero (optimizada)
# -----------------------------------------------------------------------------

# Configuración estática
DISTRIBUTIONS = {
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
MODEL_OPTIONS = [
    "gemma3:1b", "llama3.2", "phi3", "moondream", "deepseek-r1:1.5b",
]

# -----------------------------------------------------------------------------
# Caching de datos y recursos para acelerar la aplicación
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_documents_from_excel(directory: str) -> list:
    """
    Carga y transforma archivos Excel en documentos para vectorización.
    Se usa caching para evitar recargas repetitivas.
    """
    docs = []
    for fname in os.listdir(directory):
        if not fname.lower().endswith(".xlsx"):  # solo xlsx
            continue
        path = os.path.join(directory, fname)
        df = pd.read_excel(path).fillna("")
        # Construir contenido de manera vectorizada
        df = df.astype(str)
        df["content"] = df.agg(" | ".join, axis=1)
        loader = DataFrameLoader(df, page_content_column="content")
        docs.extend(loader.load())
    return docs

@st.cache_resource
def build_chain(_documents: list, model_name: str) -> RetrievalQA:
    """
    Construye el RetrievalQA con embeddings y FAISS, cached por modelo.
    """
    emb = OllamaEmbeddings(model="nomic-embed-text")
    vect = FAISS.from_documents(_documents, emb)
    retr = vect.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model=model_name)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retr,
        return_source_documents=False,
    )

# -----------------------------------------------------------------------------
# Función para generar el prompt en español
# -----------------------------------------------------------------------------

def build_prompt(selections: dict) -> str:
    lines = ["Genera un itinerario formativo que cumpla estos criterios:"]
    for cat, opts in selections.items():
        lines.append(f"- **{cat}**: {', '.join(opts)}")
    lines.append(
        "Sintetiza el programa en lista numerada, indicando duración y justificando cada recurso. "
        "Incluye los enlaces de las actividades (columna 'URL') y redacta en Español."
    )
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Interfaz Streamlit optimizada
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Constructor de Itinerarios", layout="wide")
st.title("🧩 Constructor de Itinerarios Formativos")

# Carga temprana de documentos
docs = load_documents_from_excel("../../data")

# Barra lateral: selección de modelo y criterios
st.sidebar.header("Configuración del LLM")
model = st.sidebar.selectbox(
    "Selecciona el modelo de Ollama", MODEL_OPTIONS,
    index=MODEL_OPTIONS.index("deepseek-r1:1.5b"), key="model_select"
)

st.sidebar.header("Selecciona características")
selections = {}
for cat, opts in DISTRIBUTIONS.items():
    chosen = st.sidebar.multiselect(cat, opts)
    if chosen:
        selections[cat] = chosen

# Construcción o reutilización de la cadena
if 'qa_chain' not in st.session_state or st.session_state.model != model:
    st.session_state.qa_chain = build_chain(docs, model)
    st.session_state.model = model

# Generación de itinerario on demand
if st.sidebar.button("Generar Itinerario"):
    if not selections:
        st.warning("🚨 Selecciona al menos una característica en la barra lateral.")
    else:
        prompt = build_prompt(selections)
        st.subheader("🔍 Prompt generado")
        st.code(prompt, language="markdown")

        with st.spinner(f"Generando itinerario con {model}…"):
            raw = st.session_state.qa_chain.invoke(prompt)
        # Extraer respuesta
        content = raw.get('result') if isinstance(raw, dict) else raw
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()

        st.subheader("✍️ Respuesta del Modelo")
        st.markdown(cleaned)
