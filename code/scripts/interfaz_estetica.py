import os
import re
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import DataFrameLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ----------------------------------------------------------------------------- 
# Configuración de la página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Constructor de Itinerarios",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ INYECTAR CSS PARA DETALLES DE ESTILO ------------------
custom_css = """
<style>
/* 1. Títulos y encabezados en violeta oscuro (#420078) */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #420078 !important;
}

/* 2. Botones violetas con sombra (#6247aa) */
/* Solo dentro del sidebar */
[data-testid="stSidebar"] .stButton > button {
    border-radius: 8px;
    box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.15);
    background-color: #6247aa !important;
    color: white !important;
    border: none;
}
/* Solo en el área principal */
[data-testid="stAppViewContainer"] .stButton > button {
    border-radius: 8px;
    background-color: #6247aa !important;
    color: white !important;
}

/* 3. Fondo degradado en la barra lateral (de #6247aa a #dec9e9) */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6247aa, #dec9e9) !important;
}

/* 4. Borde para selectbox y multiselect (#6247aa) */
/* Solo dentro del sidebar */
[data-testid="stSidebar"] .stSelectbox .css-1dg5wwe,
[data-testid="stSidebar"] .stMultiselect .css-1dg5wwe {
    border: 1px solid #6247aa !important;
    border-radius: 4px !important;
}
/* Solo dentro del área principal */
[data-testid="stAppViewContainer"] .stSelectbox .css-1dg5wwe,
[data-testid="stAppViewContainer"] .stMultiselect .css-1dg5wwe {
    border: 1px solid #6247aa !important;
    border-radius: 4px !important;
}

/* 5. Input de texto y textarea, solo dentro del sidebar */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    border: 1px solid #6247aa !important;
    border-radius: 4px !important;
}

/* 6. Checkbox y radio solo en sidebar */
[data-testid="stSidebar"] .stCheckbox input[type="checkbox"],
[data-testid="stSidebar"] .stRadio input[type="radio"] {
    accent-color: #6247aa; /* color de marca al seleccionarse */
}

/* 7. Estilizar bloques de código en la app principal */
[data-testid="stAppViewContainer"] .stCodeBlock pre {
    background-color: #dec9e9 !important;
    border-radius: 4px;
    padding: 8px !important;
}

/* 8. Expander (título y contenido) */
/* Título del expander solo en la zona principal */
[data-testid="stAppViewContainer"] .stExpanderHeader {
    background-color: #6247aa;
    color: white;
    padding: 8px;
    border-radius: 4px 4px 0 0;
}
/* Contenido del expander */
[data-testid="stAppViewContainer"] .stExpanderContent {
    background-color: #f5f4ff;
    border-radius: 0 0 4px 4px;
    padding: 12px;
}

/* 9. Contenedores (element-container) en el sidebar */
/* Aplica fondo pastel y margenes específicos solo a estos contenedores */
[data-testid="stSidebar"] .stElementContainer.element-container {
    background-color: #dec9e9 !important;
    border-radius: 8px;
    padding-top: 4px !important;    /* menos arriba */
    padding-right: 12px !important;
    padding-bottom: 16px !important; /* más abajo */
    padding-left: 12px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
}

/* 10. Anular padding/margin extra en headers dentro del sidebar */
[data-testid="stSidebar"] .stHeader {
    padding: 0 !important;
    margin: 0 !important;
}

/* 12. Ajustes adicionales para evitar que un componente “corrija” al otro */
[data-testid="stAppViewContainer"] .stElementContainer.element-container {
    background-color: transparent !important;
    box-shadow: none !important;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
# ------------------------------------------------------------------------------

# ----------------------------------------------------------------------------- 
# Configuración estática general
# -----------------------------------------------------------------------------
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
    docs = []
    for fname in os.listdir(directory):
        if not fname.lower().endswith(".xlsx"):
            continue
        path = os.path.join(directory, fname)
        df = pd.read_excel(path).fillna("")
        df = df.astype(str)
        df["content"] = df.agg(" | ".join, axis=1)
        loader = DataFrameLoader(df, page_content_column="content")
        docs.extend(loader.load())
    return docs

@st.cache_resource
def build_chain(_documents: list, model_name: str) -> RetrievalQA:
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
    lines = ["Genera un itinerario formativo en español que cumpla estos criterios:"]
    for cat, opts in selections.items():
        lines.append(f"- **{cat}**: {', '.join(opts)}")
    lines.append(
        "* Sintetiza el programa en lista numerada, indicando duración y justificando cada recurso."
    )
    lines.append(
        "* Incluye los enlaces de las actividades (columna 'URL')."
    )
    return "\n".join(lines)

st.title("🧩 Constructor de Itinerarios Formativos")

with st.expander("ℹ️ ¿Qué hace esta aplicación?"):
    st.write(
        """
        Esta herramienta genera automáticamente itinerarios formativos para Ciencias de la Computación,
        basándose en criterios seleccionados (tipo de recurso, competencias clave, áreas de conocimiento, etc.).
        Se utiliza un índice vectorial (FAISS) y un modelo de Ollama para ofrecer un programa detallado
        con enlaces y justificaciones.  
        """
    )

# Carga temprana de documentos
docs = load_documents_from_excel("./data")

# ----------------------- BARRA LATERAL -----------------------
st.sidebar.header("🔧 Configuración del LLM")
model = st.sidebar.selectbox(
    "🤖 Selecciona el modelo de Ollama",
    MODEL_OPTIONS,
    index=MODEL_OPTIONS.index("gemma3:1b"),
    key="model_select",
    help="Elige el modelo de Ollama. Modelos más grandes tardan más, pero suelen generar respuestas más completas."
)

st.sidebar.header("✨ Selecciona características")
selections = {}
for cat, opts in DISTRIBUTIONS.items():
    chosen = st.sidebar.multiselect(
        f"{cat} 📚",
        opts,
        help=f"Filtra por categoría '{cat}'"
    )
    if chosen:
        selections[cat] = chosen

# Validar si el botón debe habilitarse
if len(selections) == 0:
    st.sidebar.error("Selecciona al menos una característica para habilitar el botón.")
    generate_button = st.sidebar.button("Generar Itinerario", disabled=True)
else:
    generate_button = st.sidebar.button("Generar Itinerario", disabled=False)

# Construcción o reutilización de la cadena (caché por modelo)
if 'qa_chain' not in st.session_state or st.session_state.model != model:
    st.session_state.qa_chain = build_chain(docs, model)
    st.session_state.model = model

# ------------------------ GENERAR ITINERARIO ------------------------
if generate_button:
    prompt = build_prompt(selections)

    # Mostrar resumen de selecciones

    # 1) Construir la lista de líneas Markdown
    summary_lines = ["* **📋 Resumen de selecciones:**"]
    for cat, opts in selections.items():
        summary_lines.append(f"    * **{cat}:** {', '.join(opts)}")

    # 2) Unirlas en una sola string
    summary_md = "\n".join(summary_lines)

    # 3) Renderizar la string Markdown de golpe
    st.markdown(summary_md)

    # Barra de progreso en la barra lateral
    total_steps = 3
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # Paso 1: Preparar prompt
    status_text.text("1/3 – Preparando prompt...")
    progress_bar.progress(1 / total_steps)
    # (podrías hacer aquí alguna precarga extra si hiciera falta)

    # Paso 2: Invocar al LLM
    status_text.text("2/3 – Enviando consulta al modelo...")
    progress_bar.progress(2 / total_steps)
    raw = st.session_state.qa_chain.invoke(prompt)

    # Paso 3: Procesar y mostrar
    status_text.text("3/3 – Procesando respuesta...")
    progress_bar.progress(3 / total_steps)
    content = raw.get('result') if isinstance(raw, dict) else raw
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    status_text.empty()
    progress_bar.empty()

    # Mostrar prompt y resultado lado a lado
    col1, col2 = st.columns([1, 2], gap="medium")
    with col1:
        st.subheader("🔍 Prompt generado")
        st.markdown(prompt)
    with col2:
        st.subheader("✍️ Respuesta del modelo")
        st.markdown(cleaned)