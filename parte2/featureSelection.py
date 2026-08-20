"""
Parte 2 - Ejercicio 3: Seleccion y construccion de variables predictoras.

Define el conjunto de predictores a partir del dataset con variable
respuesta (parte2/resultados/dataset_con_respuesta.csv), excluyendo las
variables identificadas en el Ejercicio 2.5 como fuente de fuga de
informacion (cyano_index, B04, B05).
"""

import os
import pandas as pd

RUTA_ENTRADA = "parte2/resultados/dataset_con_respuesta.csv"
CARPETA_TABLAS = "parte2/resultados/tablas"

# ============================================================
# 3.1 / 3.2 CONJUNTO DE VARIABLES PREDICTORAS
# ============================================================
# Se excluyen explicitamente: cyano_index (variable usada para construir
# la respuesta), B04 y B05 (bandas usadas directamente en el calculo de
# NDCI/cyano_index, ver Ejercicio 2.5), y las columnas de identificacion
# no predictivas (lon, lat, lago, fecha) que se conservan solo como
# metadatos para la validacion espacial/temporal y los experimentos de
# generalizacion entre lagos (Ejercicios 6 y 7).

DESCRIPCION_PREDICTORES = {
    "B02": "Banda azul (490 nm). Sensible a dispersion por particulas "
           "finas en suspension y turbidez del agua; ayuda a distinguir "
           "aguas claras de aguas con alta carga de solidos/algas.",
    "B03": "Banda verde (560 nm). Es la banda mas sensible a la "
           "reflectancia de la clorofila-a en el agua; valores altos de "
           "reflectancia verde suelen asociarse a mayor biomasa algal.",
    "B08": "Banda infrarroja cercana (NIR, 842 nm). El agua absorbe "
           "fuertemente el NIR, pero la presencia de biomasa algal o "
           "material organico en la superficie aumenta la reflectancia "
           "NIR respecto a agua limpia; util para detectar acumulaciones "
           "superficiales densas.",
    "B11": "Banda infrarroja de onda corta (SWIR1, 1610 nm). Aporta "
           "informacion sobre contenido de humedad/materia organica y "
           "ayuda a discriminar agua de vegetacion flotante o sedimentos.",
    "ndvi": "Indice de vegetacion de diferencia normalizada. Aunque fue "
            "disenado para vegetacion terrestre, sobre cuerpos de agua "
            "valores de NDVI menos negativos/positivos pueden indicar "
            "vegetacion acuatica flotante o acumulaciones algales "
            "superficiales densas.",
    "ndwi": "Indice de agua de diferencia normalizada. Indica el grado "
            "de 'pureza' espectral del agua; caidas en NDWI dentro del "
            "cuerpo de agua pueden reflejar interferencia de material "
            "en suspension, incluidas floraciones algales.",
}

COLUMNAS_METADATA = ["lon", "lat", "lago", "fecha"]
COLUMNAS_EXCLUIDAS_FUGA = ["cyano_index", "B04", "B05"]
COLUMNA_RESPUESTA = "alta_cianobacteria"


def cargar_dataset():
    return pd.read_csv(RUTA_ENTRADA, parse_dates=["fecha"])


def construir_conjunto_predictores(df):
    columnas_predictoras = [
        c for c in df.columns
        if c not in COLUMNAS_METADATA + COLUMNAS_EXCLUIDAS_FUGA + [COLUMNA_RESPUESTA]
    ]
    return columnas_predictoras


if __name__ == "__main__":
    os.makedirs(CARPETA_TABLAS, exist_ok=True)

    df = cargar_dataset()
    predictores = construir_conjunto_predictores(df)

    print("\n--- Variables predictoras seleccionadas ---")
    for p in predictores:
        print(f"- {p}: {DESCRIPCION_PREDICTORES.get(p, '(banda espectral cruda)')}")

    print("\n--- Variables excluidas por posible fuga de informacion ---")
    for c in COLUMNAS_EXCLUIDAS_FUGA:
        print(f"- {c}")

    tabla_predictores = pd.DataFrame({
        "variable": predictores,
        "descripcion": [DESCRIPCION_PREDICTORES.get(p, "banda espectral cruda de Sentinel-2") for p in predictores],
    })
    tabla_predictores.to_csv(os.path.join(CARPETA_TABLAS, "predictores_seleccionados.csv"), index=False)
    print(f"\nTabla de predictores guardada -> {os.path.join(CARPETA_TABLAS, 'predictores_seleccionados.csv')}")
