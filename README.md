# Generador de itinerarios formativos personalizados con IA

Este proyecto desarrolla una herramienta basada en inteligencia artificial para la generación automática de itinerarios formativos adaptados a diferentes niveles educativos, competencias y valores, utilizando una base de recursos clasificados mediante técnicas semiautomáticas.

---

## 📁 Estructura del repositorio

```bash
.
├── LICENSE
├── README.md
├── requirements.txt
├── code/                                                                       # Código fuente del proyecto
│   ├── notebooks/                                                              # Notebooks para análisis y procesamiento de datos
│   │   ├── enlaces.ipynb                                                       # Asignación automática de URLs a recursos
│   │   ├── enlaces.pdf                                                         # Exportación PDF del anterior
│   │   ├── recursos_estadisticas.ipynb                                         # Análisis estadístico de recursos educativos
│   │   └── recursos_estadisticas.pdf                                           # Exportación PDF del anterior
│   └── scripts/                                                                # Scripts ejecutables
│       ├── chat.py                                                             # Asistente conversacional con recuperación semántica
│       ├── interfaz.py                                                         # Interfaz Streamlit inicial para generación de itinerarios
│       ├── interfaz_estética.py                                                # Versión estética de la herramienta (RECOMENDACIÓN)
│       ├── interfaz_optimizado.py                                              # Versión optimizada con caché y selección de modelo
│       ├── prompts.py                                                          # Generador masivo de respuestas en Markdown
│       ├── revision_sistematica.py                                             # Comparador entre revisiones y clasificación oficial
│       ├── seleccion.py                                                        # Interfaz por terminal para crear prompts educativos
│       ├── test.py                                                             # Prueba básica de conversación con LLM (modo consola)
│       └── tiempos.py                                                          # Medición de tiempos de respuesta por modelo
├── data/
│   └── recursos_clasificados.xlsx                                              # Clasificación oficial de recursos (base de datos)
├── docs/                                                                       # Documentación generada con Sphinx
├── documents/                                                                  # Documentos estáticos y fijos para análisis
│   ├── answers/                                                                # Respuestas generadas por distintos modelos en PDF
│   │   ├── respuestas_coze.pdf
│   │   ├── respuestas_deepseek.pdf
│   │   ├── respuestas_gemma3.pdf
│   │   ├── respuestas_googlecloud.pdf
│   │   ├── respuestas_llama32.pdf
│   │   └── respuestas_phi3.pdf
│   ├── evaluacion_modelos.pdf                                                  # Evaluación cualitativa comparativa entre modelos
│   ├── evaluacion_sistematica.csv                                              # Validación de etiquetas por categoría
│   ├── plots/                                                                  # Gráficos estadísticos del dataset de recursos
│   │   └── *.png                                                               # Categorías como duration, key_competences, etc.
│   ├── resources/                                                              # Bases de datos originales de recursos educativos
│   │   ├── ceibal.xlsx (...)
│   │   ├── enlaces.xlsx                                                        # Clasificación cruda realizada por Elicit (sin las columnas de explicación)
│   │   └── output.xlsx                                                         # Resultado de la asignación automática de enlaces
│   ├── revision_sistematica_1.xlsx                                             # Clasificación realizada por la revisión sistemática 1
│   ├── revision_sistematica_2.xlsx                                             # Clasificación realizada por la revisión sistemática 2
│   └── rubrica.pdf                                                             # Rúbrica utilizada para evaluar las respuestas de modelos
├── results/                                                                    # Resultados generados por los scripts
│   ├── answers/                                                                # Respuestas en Markdown por modelo
│   │   ├── respuestas_deepseek.md
│   │   ├── respuestas_gemma.md
│   │   ├── respuestas_llama32.md
│   │   ├── respuestas_moondream.md
│   │   └── respuestas_phi3.md
│   ├── resultado_validacion.xlsx                                               # Informe de validación sistemática por recurso
│   ├── resultados_tiempo.csv                                                   # Tiempos de respuesta de los modelos (formato tabla)
│   └── resultados_tiempo.md                                                    # Tiempos de respuesta (formato legible Markdown)
```

## 🚀 Instalación

### Requisitos previos

- Sistema operativo compatible (Windows preferiblemente)
- [Ollama](https://ollama.com/) instalado y funcionando localmente
- Python 3.10 o superior

### Instrucciones paso a paso

1. **Instala Ollama**  
   
Visita [https://ollama.com](https://ollama.com) y descarga el instalador para tu sistema operativo. Sigue las instrucciones oficiales para completar la instalación.

2. **Prepara un entorno virtual de Python**

```bash
python -m venv .venv
```

3. **Activa el entorno virtual**

   * En Windows:

```bash
.venv\Scripts\activate
```

4. **Descarga los modelos de Ollama que vas a usar**
   
Puedes hacerlo con el siguiente comando (repite para cada modelo necesario):

```bash
ollama pull nomic-embed-text
ollama pull gemma:3b
ollama pull phi3
ollama pull llama3
ollama pull deepseek
ollama pull moondream
```

5. **Instala las dependencias del proyecto**

```bash
pip install -r requirements.txt
```

6. **Ejecuta la interfaz web con Streamlit**

```bash
streamlit run code/scripts/interfaz_estetica.py
```

### 📦 Modelos utilizados en el proyecto

| Modelo               | Tamaño aproximado | Características principales                                        | Requisitos recomendados     |
|----------------------|-------------------|--------------------------------------------------------------------|------------------------------|
| `phi3`               | ~1.2 GB           | Muy rápido y ligero, ideal para tareas básicas                     | CPU moderna, ≥ 8 GB de RAM   |
| `llama3.2`           | ~4.7 GB           | Buen equilibrio entre fluidez y contexto, multilingüe              | 12–16 GB de RAM (mejor con GPU) |
| `gemma3:1b`          | ~2.8 GB           | Modelo multilingüe eficiente, buen rendimiento en tareas educativas| CPU o GPU, ≥ 8 GB de RAM     |
| `deepseek-r1:1.5b`   | ~3.1 GB           | Robusto en razonamiento, algo más lento, útil en prompts complejos | CPU potente o GPU, ≥ 12 GB de RAM |
| `moondream`          | ~800 MB           | Muy ligero, ideal para respuestas rápidas y entornos limitados     | CPU ligera, ≥ 6 GB de RAM    |

> ⚠️ Algunos modelos pueden necesitar aceleración por GPU para obtener tiempos de respuesta óptimos. El sistema ha sido probado principalmente en CPU con 8GB de RAM.

---

## 📄 Licencia

Este proyecto está licenciado bajo:

> **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**  
> [Ver licencia completa](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## ✍️ Autora

**Esther Medina Quintero**  
_TFG | Grado en Ingeniería Informática | Universidad de La Laguna_