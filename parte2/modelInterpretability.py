"""
Parte 2 - Ejercicio 8: Interpretacion y explicabilidad del modelo.

Analiza el modelo con mejor desempeno segun el criterio ambiental del
Ejercicio 5.3 (XGBoost, mejor F2/recall) mediante importancia de
variables y SHAP.
"""

import os
import joblib
import numpy as np
import pandas as pd

RUTA_DATOS = "parte2/resultados/dataset_features.csv"
CARPETA_MODELOS = "parte2/resultados/modelos"
CARPETA_TABLAS = "parte2/resultados/tablas"
CARPETA_FIGURAS = "parte2/resultados/figuras/interpretabilidad"

RANDOM_STATE = 42
TEST_SIZE = 0.30

PREDICTORES = [
    "ndvi", "ndwi", "B02", "B03", "B08", "B11",
    "ratio_verde_azul", "brillo_superficial", "indice_turbidez_verde_swir",
]
COLUMNA_RESPUESTA = "alta_cianobacteria"
MODELO_SELECCIONADO = "xgboost"  # ver parte2/resultados/tablas/mejor_modelo_ambiental.txt


def cargar_datos_y_modelo():
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(RUTA_DATOS, parse_dates=["fecha"])
    X = df[PREDICTORES]
    y = df[COLUMNA_RESPUESTA]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    modelo = joblib.load(os.path.join(CARPETA_MODELOS, f"{MODELO_SELECCIONADO}_final.joblib"))
    return modelo, X_test, y_test


# ============================================================
# 8.1 IMPORTANCIA GLOBAL DE VARIABLES
# ============================================================

def importancia_global(modelo, X_test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(CARPETA_FIGURAS, exist_ok=True)

    importancias = pd.Series(modelo.feature_importances_, index=PREDICTORES).sort_values(ascending=False)

    print("\n--- Importancia global de variables (XGBoost, importance_type por defecto) ---")
    print(importancias.to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    importancias.sort_values().plot(kind="barh", ax=ax, color="#2b7a78")
    ax.set_xlabel("Importancia")
    ax.set_title(f"Importancia global de variables - {MODELO_SELECCIONADO}")
    fig.tight_layout()

    ruta = os.path.join(CARPETA_FIGURAS, "importancia_global_variables.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico de importancia guardado -> {ruta}")

    importancias.reset_index().rename(columns={"index": "variable", 0: "importancia"}).to_csv(
        os.path.join(CARPETA_TABLAS, "importancia_global_variables.csv"), index=False
    )

    return importancias


if __name__ == "__main__":
    os.makedirs(CARPETA_TABLAS, exist_ok=True)
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)

    modelo, X_test, y_test = cargar_datos_y_modelo()
    importancia_global(modelo, X_test)
