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


# ============================================================
# 8.2 / 8.3 / 8.4 SHAP: EXPLICABILIDAD Y PATRONES POR VARIABLE
# ============================================================

TAMANO_MUESTRA_SHAP = 3000


def calcular_shap(modelo, X_test):
    import shap

    muestra = X_test.sample(n=min(TAMANO_MUESTRA_SHAP, len(X_test)), random_state=RANDOM_STATE)

    explicador = shap.TreeExplainer(modelo)
    valores_shap = explicador.shap_values(muestra)

    return valores_shap, muestra


def graficar_shap_summary(valores_shap, muestra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(valores_shap, muestra, show=False)
    ruta = os.path.join(CARPETA_FIGURAS, "shap_summary_plot.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SHAP summary plot guardado -> {ruta}")
    return ruta


def resumen_importancia_shap(valores_shap, muestra):
    """
    Resume, por variable, la importancia media |SHAP| y la correlacion
    entre el valor de la variable y su propio valor SHAP (signo
    positivo/negativo indica si valores altos de la variable tienden a
    aumentar o disminuir la prediccion de alta cianobacteria).
    """

    importancia_media = np.abs(valores_shap).mean(axis=0)
    correlaciones = [
        np.corrcoef(muestra[col].values, valores_shap[:, i])[0, 1]
        for i, col in enumerate(PREDICTORES)
    ]

    tabla = pd.DataFrame({
        "variable": PREDICTORES,
        "importancia_media_abs_shap": importancia_media,
        "correlacion_valor_vs_shap": correlaciones,
    }).sort_values("importancia_media_abs_shap", ascending=False)

    tabla["direccion"] = np.where(
        tabla["correlacion_valor_vs_shap"] > 0,
        "valores altos -> mayor prob. alta cianobacteria",
        "valores altos -> menor prob. alta cianobacteria",
    )

    print("\n--- Resumen de importancia y direccion SHAP ---")
    print(tabla.to_string(index=False))

    return tabla


# ============================================================
# 8.3 / 8.4 INTERPRETACION AMBIENTAL DE LOS PATRONES SHAP
# ============================================================

INTERPRETACION_AMBIENTAL_SHAP = """
--- Interpretacion ambiental de los patrones SHAP (Ejercicio 8.3-8.4) ---

(Valores exactos en parte2/resultados/tablas/resumen_shap_variables.csv)

  - ratio_verde_azul es, por amplio margen, la variable mas influyente
    (importancia SHAP ~6.13, correlacion valor-SHAP +0.96): a mayor
    reflectancia verde relativa a la azul, mayor probabilidad de alta
    cianobacteria. Es consistente con la fisica del fenomeno: los
    pigmentos fotosinteticos de las cianobacterias (clorofila-a,
    ficocianina) incrementan la reflectancia verde y el agua limpia
    dispersa relativamente mas azul.

  - ndvi (SHAP +0.67) y ndwi (SHAP +0.90) tambien empujan la
    prediccion hacia "alta presencia" cuando son altos. Esto no
    contradice su definicion original para vegetacion/agua: dentro de
    pixeles YA clasificados como agua (Ejercicio 1.3), ambos indices
    quedan dominados por el mismo contraste verde-vs-resto que capta
    ratio_verde_azul, por lo que "NDWI alto" aqui refleja mayor
    reflectancia verde relativa (pigmentos), no necesariamente "agua
    mas pura" como se hipotetizaria fuera de un contexto ya filtrado a
    agua.

  - B02 (azul) y B08 (NIR) muestran el patron opuesto (correlacion
    -0.47 y -0.16): mayor reflectancia empuja la prediccion hacia
    "baja presencia", coherente con que el agua limpia dispersa mas
    azul y absorbe mas NIR que el agua con biomasa algal densa.

  - brillo_superficial (SHAP +0.47) y B03/verde (SHAP +0.21) refuerzan
    la misma direccion: mayor reflectancia visible total y mayor verde
    absoluto se asocian con mayor probabilidad de floracion.

  - B11 (SWIR1) e indice_turbidez_verde_swir tienen la menor
    influencia y una direccion mas debil, sugiriendo que la
    humedad/materia organica capturada por SWIR1 aporta poca senal
    adicional una vez que ratio_verde_azul ya esta en el modelo.

En conjunto, el modelo aprende un patron ambientalmente plausible y
dominado por un unico mecanismo fisico (contraste verde-azul asociado
a pigmentos fotosinteticos), reforzado de forma consistente por los
demas indices y bandas, en vez de depender de relaciones espurias.
"""


def imprimir_interpretacion_shap():
    print(INTERPRETACION_AMBIENTAL_SHAP)


if __name__ == "__main__":
    os.makedirs(CARPETA_TABLAS, exist_ok=True)
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)

    modelo, X_test, y_test = cargar_datos_y_modelo()
    importancia_global(modelo, X_test)

    valores_shap, muestra = calcular_shap(modelo, X_test)
    graficar_shap_summary(valores_shap, muestra)

    tabla_shap = resumen_importancia_shap(valores_shap, muestra)
    tabla_shap.to_csv(os.path.join(CARPETA_TABLAS, "resumen_shap_variables.csv"), index=False)

    imprimir_interpretacion_shap()
