# Generador de Itinerarios Formativos Personalizados con IA

Este proyecto desarrolla una herramienta basada en inteligencia artificial para la generación automática de itinerarios formativos adaptados a diferentes niveles educativos, competencias y valores, utilizando una base de recursos clasificados mediante técnicas semiautomáticas.

---

## 🎯 Objetivo

Facilitar la personalización del aprendizaje en Ciencias de la Computación mediante:

- Clasificación estructurada de recursos educativos abiertos.
- Generación de itinerarios didácticos personalizados con modelos de lenguaje (LLMs).
- Interfaz accesible para su uso por profesorado no técnico.
- Evaluación rigurosa de la calidad pedagógica de las respuestas.

---

## 🧱 Arquitectura del sistema

El sistema se organiza en los siguientes módulos:

- **`📂 data/`**: Contiene los datos clasificados y procesados para la generación de itinerarios.
- **`📂 docs/`**: Documentación del proyecto generada con Sphinx.
- **`📂 documents/`**: Recursos y evaluaciones utilizadas en el desarrollo del sistema.
- **`📂 results/`**: Resultados obtenidos tras la evaluación de los modelos y tiempos de ejecución.
- **`📂 code/scripts/`**: Scripts principales para la generación de itinerarios, interfaz y evaluación.
- **`📂 code/notebooks/`**: Notebooks para análisis exploratorio y estadístico.

Archivos principales:

- **`LICENSE`**: Información sobre la licencia del proyecto.
- **`README.md`**: Este archivo, con información general del proyecto.
- **`requirements.txt`**: Dependencias necesarias para ejecutar el sistema.

---

## 🚀 Instalación

### Requisitos previos

- Python 3.10 o superior.
- [Ollama](https://ollama.com/) instalado y configurado localmente.
- Modelos ligeros compatibles (`phi3`, `gemma`, `llama3`, etc.).

### Instrucciones

1. Clona este repositorio:

    ```bash
    git clone https://github.com/tu-usuario/tfg-itinerarios-ia.git
    cd tfg-itinerarios-ia
    ```

2. Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

3. Lanza la interfaz web:

    ```bash
    streamlit run interfaz_optimizado.py
    ```

---

## 🤖 Modelos utilizados

Los siguientes modelos se integran con Ollama y se ejecutan en local con un mínimo de 8 GB de RAM:

- **`phi3`**
- **`llama3.2`**
- **`gemma3:1b`**
- **`deepseek-r1:1.5b`**
- **`moondream`**

---

## 📊 Evaluación del sistema

- **Validación de clasificación:** Revisión sistemática con un 70% de aciertos en la clasificación de recursos.
- **Evaluación de itinerarios:** Aplicación de una rúbrica en 6 escenarios simulados.
- **Modelo más robusto:** Coze (3.6/4), seguido de Gemma y Llama.
- **Criterios valorados:** Tiempo de respuesta, precisión, formato, relevancia y robustez.

---

## 📄 Licencia

Este proyecto está licenciado bajo:

> **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**  
> [Ver licencia completa](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## ✍️ Autora

**Esther Medina Quintero**  
_TFG | Grado en Ingeniería Informática | Universidad de La Laguna_