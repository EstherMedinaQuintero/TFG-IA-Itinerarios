# Generador de Itinerarios Formativos Personalizados con IA

Este proyecto desarrolla una herramienta basada en inteligencia artificial para la generación automática de itinerarios formativos adaptados a diferentes niveles educativos, competencias y valores, a partir de una base de recursos clasificados mediante técnicas semiautomáticas.

## 🎯 Objetivo

Facilitar la personalización del aprendizaje en Ciencias de la Computación mediante:

- Clasificación estructurada de recursos educativos abiertos.
- Generación de itinerarios didácticos personalizados con LLMs.
- Interfaz accesible para su uso por profesorado no técnico.
- Evaluación rigurosa de la calidad pedagógica de las respuestas.

## 🧱 Arquitectura del sistema

El sistema se organiza en los siguientes módulos:

- `📂 data/`: (...)
- `📂 docs/`(...)
- `📂 evaluacion/`(...)
- `📂 recursos_raw/`(...)
- `📂 resultados/`(...)
- `📂 scripts/`(...)
- LICENSE.md
- README.md
- requirements.md

## 🚀 Instalación

### Requisitos previos

- Python 3.10 o superior
- [Ollama](https://ollama.com/) instalado y configurado localmente
- Modelos ligeros compatibles (`phi3`, `gemma`, `llama3`, etc.)

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

## 🤖 Modelos utilizados

Los siguientes modelos se integran con Ollama y se ejecutan en local con 8 GB de RAM:

- `phi3`
- `llama3.2`
- `gemma3:1b`
- `deepseek-r1:1.5b`
- `moondream`

## 📊 Evaluación del sistema

- **Validación de clasificación:** Revisión sistemática (70% de aciertos).
- **Evaluación de itinerarios:** Rúbrica con 6 escenarios simulados.
- **Modelo más robusto:** Coze (3.6/4), seguido de Gemma y Llama.
- **Criterios valorados:** Tiempo, precisión, formato, relevancia y robustez.

## 📄 Licencia

Este proyecto está licenciado bajo:

> **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**  
> [Ver licencia completa](https://creativecommons.org/licenses/by-nc-nd/4.0/)

**Autora: Esther Quintero**  
_TFG | Grado en Ingeniería Informática | Universidad de La Laguna_