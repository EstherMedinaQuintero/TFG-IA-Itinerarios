import pandas as pd
import os

def contiene(texto, lista):
    return any(palabra.lower() in texto.lower() for palabra in lista if isinstance(texto, str))

def comparar_filas(fila_rev, fila_clas, columnas_revision):
    resultado = {"Filename": fila_rev["Filename"]}
    correctos = 0
    totales = 0

    # Mapeo ampliado de campos con comprobaciones dinámicas
    mapeo = {
        "Addresses_Patterns": lambda: contiene(fila_clas.get("Ct skills", ""), ["Patrones"]),
        "Addresses_LogicalReasoning": lambda: contiene(fila_clas.get("Ct skills", ""), ["Razonamiento Lógico"]),
        "Addresses_AlgorithmicThinkingProgramming": lambda: contiene(fila_clas.get("Ct skills", ""), ["Pensamiento Algorítmico", "Programación"]),
        "Addresses_Evaluation": lambda: contiene(fila_clas.get("Ct skills", ""), ["Evaluación"]),
        "Mentions_NetworksInternet": lambda: contiene(fila_clas.get("Cc concepts", ""), ["Redes e Internet"]),
        "Mentions_ComputingImpact": lambda: contiene(fila_clas.get("Cc concepts", ""), ["Impacto de la Computación"]),
        "Type_PluggedActivity": lambda: contiene(fila_clas.get("Resource type", ""), ["enchufada"]),
        "Type_UnpluggedActivity": lambda: contiene(fila_clas.get("Resource type", ""), ["desenchufada"]),
        "Type_VisualProgramming": lambda: contiene(fila_clas.get("Resource type", ""), ["visual"]),
        "Type_Course": lambda: contiene(fila_clas.get("Resource type", ""), ["Curso"]),
        "Type_Video": lambda: contiene(fila_clas.get("Resource type", ""), ["Vídeo"]),
        "Type_TextualProgramming": lambda: contiene(fila_clas.get("Resource type", ""), ["textual"]),
        "Type_PhysicalDevices": lambda: contiene(fila_clas.get("Cc concepts", ""), ["Dispositivos", "Robótica"]),
        "Duration_QuickActivity": lambda: contiene(fila_clas.get("Duration", ""), ["Rápida"]),
        "Duration_TwoMonths": lambda: contiene(fila_clas.get("Duration", ""), ["2 Meses"]),
        "Duration_ThreePlusMonths": lambda: contiene(fila_clas.get("Duration", ""), ["3 Meses", "Más de 3 Meses"]),
        "Fits_EducacionInfantil": lambda: contiene(fila_clas.get("School years", ""), ["Infantil"]),
        "Fits_CicloEP": lambda: contiene(fila_clas.get("School years", ""), ["Ciclo EP"]),
        "Fits_ESO": lambda: contiene(fila_clas.get("School years", ""), ["ESO"]),
        "Fits_Bachillerato": lambda: contiene(fila_clas.get("School years", ""), ["Bachillerato"]),
        "Fits_Mathematics": lambda: contiene(fila_clas.get("Knowledege areas", ""), ["Matemáticas"]),
        "Promotes_Teamwork": lambda: contiene(fila_clas.get("Values", ""), ["Trabajo en equipo"]),
        "Promotes_CreativityScientificSpirit": lambda: contiene(fila_clas.get("Values", ""), ["creatividad", "espíritu científico"]),
        "Promotes_GoodUseICT": lambda: contiene(fila_clas.get("Values", ""), ["Buen uso de las TIC"]),
        "Targets_LinguisticCommunication": lambda: contiene(fila_clas.get("Key competences", ""), ["Lingüística"]),
        "Targets_DigitalCompetence": lambda: contiene(fila_clas.get("Key competences", ""), ["Digital"]),
        "Targets_MathScienceTechCompetence": lambda: contiene(fila_clas.get("Key competences", ""), ["Matemática", "Tecnología"]),
        "Targets_LearningToLearn": lambda: contiene(fila_clas.get("Key competences", ""), ["Aprender a Aprender"]),
        "Targets_SocialAndCivic": lambda: contiene(fila_clas.get("Key competences", ""), ["Sociales", "Cívicas"]),
        "Targets_Plurilingual": lambda: contiene(fila_clas.get("Key competences", ""), ["Plurilingüe"]),
        "Available_English": lambda: contiene(fila_clas.get("Languages", ""), ["Inglés"])
    }

    for campo, funcion in mapeo.items():
        if campo in columnas_revision:
            valor_esperado = fila_rev.get(campo, "").strip().lower()
            valor_obtenido = funcion()
            if valor_esperado in ["yes", "no"]:
                esperado = valor_esperado == "yes"
                correcto = esperado == valor_obtenido
                resultado[campo] = "✔️" if correcto else "❌"
                totales += 1
                correctos += int(correcto)
            elif valor_esperado == "maybe":
                resultado[campo] = "maybe"
            else:
                resultado[campo] = "no info"

    resultado["Aciertos"] = correctos
    resultado["Total Comparables"] = totales
    resultado["% Acierto"] = round((correctos / totales) * 100, 2) if totales else 0
    return resultado

def validar_clasificacion(revision_df, clasificacion_df):
    resultados = []
    columnas_revision = revision_df.columns

    for _, fila_rev in revision_df.iterrows():
        archivo = fila_rev["Filename"]
        fila_clas_match = clasificacion_df[clasificacion_df["Filename"] == archivo]

        if not fila_clas_match.empty:
            fila_clas = fila_clas_match.iloc[0]
            resultados.append(comparar_filas(fila_rev, fila_clas, columnas_revision))
        else:
            resultados.append({
                "Filename": archivo,
                "Error": "Archivo no encontrado en clasificación",
                "Aciertos": 0,
                "Total Comparables": 0,
                "% Acierto": 0
            })

    return pd.DataFrame(resultados)

def main():
    carpeta_revision = "revision"
    archivo_clasificacion = "./data/OutputEditado.xlsx"
    carpeta_salida = "resultados"

    os.makedirs(carpeta_salida, exist_ok=True)

    try:
        clasificacion_df = pd.read_excel(archivo_clasificacion)
        resumen_global = []

        for archivo in os.listdir(carpeta_revision):
            if archivo.endswith(".xlsx"):
                path = os.path.join(carpeta_revision, archivo)
                revision_df = pd.read_excel(path)
                resultado = validar_clasificacion(revision_df, clasificacion_df)

                # Guardar resultados por archivo
                salida_path = os.path.join(carpeta_salida, f"resultado_{archivo}")
                resultado.to_excel(salida_path, index=False)

                # Calcular promedio de aciertos
                aciertos = resultado["Aciertos"].sum()
                total = resultado["Total Comparables"].sum()
                porcentaje = round((aciertos / total) * 100, 2) if total else 0
                resumen_global.append((archivo, aciertos, total, porcentaje))

        # Mostrar resumen en consola
        print("\n📊 Resumen de validaciones:\n")
        for archivo, aciertos, total, porcentaje in resumen_global:
            print(f"Archivo: {archivo} | Aciertos: {aciertos}/{total} | % Acierto: {porcentaje}%")

        media_total = round(sum(p for _, _, _, p in resumen_global) / len(resumen_global), 2)
        print(f"\n✅ Porcentaje medio global de acierto: {media_total}%")

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")

if __name__ == "__main__":
    main()
