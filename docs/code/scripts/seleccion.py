import os
import pandas as pd

from langchain_community.document_loaders import DataFrameLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -----------------------------------------------------------------------------
# Módulo: seleccion.py
# Descripción: Permite seleccionar criterios de recursos formativos mediante prompt
#              interactivo, genera un prompt en Español y ejecuta RetrievalQA con
#              embeddings de Ollama sobre documentos Excel.
# Autora: Esther M. Quintero
# -----------------------------------------------------------------------------

# Diccionario de opciones disponibles para generar el itinerario
DISTRIBUTIONS = {
    "Resource type": [
        "Actividad desenchufada",
        "Programación visual",
        "Actividad enchufada",
        "Curso",
        "Vídeo",
        "Dispositivos físicos",
        "Programación textual",
    ],
    "Values": [
        "Pensamiento crítico",
        "Fomento de la creatividad y del espíritu científico",
        "Trabajo en equipo",
        "Buen uso de las TIC",
        "Convivencia y Educación Cívica",
        "Atención a la diversidad",
        "Educación Ambiental y desarrollo sostenible",
        "Interculturalidad",
        "Igualdad de Género",
        "Educación para la Salud",
        "Fomento de la creatividad",
    ],
    "Key competences": [
        "Aprender a Aprender",
        "Competencia Matemática y en Ciencia y Tecnología",
        "Competencia Digital",
        "Sociales y Cívicas",
        "Comunicación Lingüística",
        "Plurilingüe",
    ],
    "Knowledge areas": [
        "Informática y Robótica",
        "Matemáticas",
        "Ética",
        "Lengua y Literatura",
        "Arte",
        "Biología y Geología",
        "Filosofía",
        "Geografía e Historia",
        "Música",
        "Deporte",
        "Física",
        "Tecnología",
        "Ciencias (Biología y Geología)",
        "Física y Química",
        "Física (dentro de Biología y Geología)",
    ],
    "Cc concepts": [
        "Algoritmia y Programación",
        "Análisis de Datos",
        "Impacto de la Computación",
        "Sistemas Informáticos",
        "Redes e Internet",
    ],
    "Ct skills": [
        "Razonamiento Lógico",
        "Descomposición",
        "Evaluación",
        "Pensamiento Algorítmico y Programación",
        "Abstracción",
        "Patrones",
    ],
    "Duration": [
        "Actividad Rápida (una sola clase)",
        "Sesión (hasta 2 horas)",
        "Semana (2-4 sesiones)",
        "Mes (5-15 sesiones)",
        "2 Meses (15-30 sesiones)",
        "Más de 3 Meses (Más de 30 sesiones)",
    ],
    "School years": [
        "1º ESO",
        "Tercer Ciclo EP",
        "2º ESO",
        "3º ESO",
        "Segundo Ciclo EP",
        "4º ESO",
        "1º Bachillerato",
        "2º Bachillerato",
        "Primer Ciclo EP",
        "Educación Infantil",
    ],
    "Languages": [
        "Inglés",
        "Español",
    ]
}


def prompt_user_selection() -> dict:
    """
    Muestra las categorías y sus opciones al usuario, permitiendo selecciones.

    Para cada categoría en DISTRIBUTIONS, imprime las opciones numeradas
    y solicita al usuario índices separados por comas. Si no introduce nada,
    pasa a la siguiente categoría.

    Returns:
        dict: Mapeo de categoría a lista de opciones seleccionadas.
    """
    selections: dict = {}

    for category, options in DISTRIBUTIONS.items():
        print(f"\nSelecciona una o varias opciones para «{category}»: ")
        for idx, option in enumerate(options, start=1):
            print(f"  {idx}. {option}")

        user_input = input(
            "Índices separados por comas (e.g. 1,3,5), o ENTER para saltar: "
        ).strip()

        if not user_input:
            continue

        chosen: list = []
        for idx_str in user_input.split(","):  # type: ignore
            try:
                idx = int(idx_str)
            except ValueError:
                continue

            if 1 <= idx <= len(options):
                chosen.append(options[idx - 1])

        if chosen:
            selections[category] = chosen

    return selections


def build_prompt(selections: dict) -> str:
    """
    Construye un prompt en Español que incluye todas las selecciones.

    Args:
        selections (dict): Mapeo de categoría a opciones seleccionadas.

    Returns:
        str: Texto completo del prompt para el modelo.
    """
    lines: list = [
        "Genera un itinerario formativo que cumpla estos criterios:"
    ]

    for category, chosen in selections.items():
        lines.append(f"- **{category}**: {', '.join(chosen)}")

    lines.append(
        "Sintetiza el programa en formato de lista numerada, "
        "indicando duración y justificando brevemente cada recurso. "
        "Incluye los enlaces de las actividades (columna 'URL') y redacta la respuesta en Español."
    )

    return "\n".join(lines)


def load_documents_from_excel(directory: str) -> list:
    """
    Carga documentos de todos los archivos Excel en un directorio.

    Args:
        directory (str): Ruta al directorio con archivos .xlsx.

    Returns:
        list: Documentos cargados listos para indexar.
    """
    print("Cargando documentos desde Excel…")
    documents: list = []

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

        loader = DataFrameLoader(
            df, page_content_column="content"
        )
        documents.extend(loader.load())

    print(f"✔ {len(documents)} documentos cargados.")
    return documents


def build_chain(documents: list) -> RetrievalQA:
    """
    Genera embeddings y prepara la cadena RetrievalQA.

    Args:
        documents (list): Documentos a indexar.

    Returns:
        RetrievalQA: Cadena configurada con Ollama y FAISS.
    """
    print("Generando embeddings y creando el índice FAISS…")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = OllamaLLM(model="moondream")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    print("✔ Cadena de recuperación lista.")
    return chain


def main() -> None:
    """
    Flujo principal: carga datos, selecciona criterios, genera prompt y pregunta.
    """
    # Cargar documentos y construir cadena
    docs = load_documents_from_excel("./data")
    chain = build_chain(docs)

    # Selección interactiva de criterios
    selections = prompt_user_selection()
    if not selections:
        print("No se seleccionó ningún criterio. Terminando.")
        return

    # Construir y mostrar prompt
    prompt = build_prompt(selections)
    print("\n== Prompt generado ==\n")
    print(prompt)

    # Ejecutar RetrievalQA y mostrar respuesta
    print("\n== Generando itinerario con el modelo ==\n")
    respuesta = chain.invoke(prompt)
    print(respuesta)


if __name__ == "__main__":
    main()
