"""
Parte 2 - Ejercicio 6: Validacion espacial.

Divide cada lago en una cuadricula regular de ~1 km x 1 km (en el
sistema de referencia WGS 84 / UTM zona 15N, EPSG:32615, que usa metros
como unidad de medida), asigna cada observacion a un bloque espacial y
visualiza los bloques generados.
"""

import os
import numpy as np
import pandas as pd
from pyproj import Transformer

RUTA_DATOS = "parte2/resultados/dataset_features.csv"
CARPETA_TABLAS = "parte2/resultados/tablas"
CARPETA_FIGURAS = "parte2/resultados/figuras/espacial"

TAMANO_BLOQUE_M = 1000  # 1 km x 1 km

TRANSFORMADOR = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)


def cargar_datos():
    return pd.read_csv(RUTA_DATOS, parse_dates=["fecha"])


# ============================================================
# 6.1 REPROYECCION Y CUADRICULA ESPACIAL
# ============================================================

def reproyectar(df):
    """
    Reproyecta lon/lat (EPSG:4326) a EPSG:32615 (WGS 84 / UTM 15N,
    metros). Se usa el mismo CRS para ambos lagos, tal como lo pide el
    enunciado, aunque Amatitlan queda cerca del limite con la zona 16N;
    esto no afecta la validez de los bloques de 1 km, solo introduce
    una distorsion metrica minima y aceptable para este proposito.
    """

    df = df.copy()
    x, y = TRANSFORMADOR.transform(df["lon"].values, df["lat"].values)
    df["x_utm"] = x
    df["y_utm"] = y
    return df


def asignar_bloques(df, tamano_m=TAMANO_BLOQUE_M):
    """
    Asigna cada observacion a un bloque de tamano_m x tamano_m metros,
    dentro de su lago (los bloques se identifican por lago + indice de
    columna/fila de la cuadricula, para no mezclar bloques de distintos
    lagos aunque coincidieran numericamente).
    """

    df = df.copy()
    df["bloque_col"] = np.floor(df["x_utm"] / tamano_m).astype(int)
    df["bloque_fila"] = np.floor(df["y_utm"] / tamano_m).astype(int)
    df["bloque_id"] = (
        df["lago"] + "_" + df["bloque_col"].astype(str) + "_" + df["bloque_fila"].astype(str)
    )
    return df


def evaluar_tamano_bloques(df):
    """
    Reporta, para cada lago, el numero de bloques con al menos una
    observacion valida y la cantidad de observaciones por bloque, para
    verificar que 1 km x 1 km produce suficientes bloques utiles para
    la validacion espacial (Ejercicio 6.1).
    """

    resumen = (
        df.groupby(["lago", "bloque_id"])
        .size()
        .reset_index(name="n_observaciones")
    )

    conteo_bloques = resumen.groupby("lago").size().rename("n_bloques")
    stats_obs = resumen.groupby("lago")["n_observaciones"].agg(["mean", "median", "min", "max"])

    tabla = pd.concat([conteo_bloques, stats_obs], axis=1).reset_index()
    tabla.columns = ["lago", "n_bloques", "obs_promedio_por_bloque", "obs_mediana_por_bloque", "obs_min_por_bloque", "obs_max_por_bloque"]

    print("\n--- Bloques espaciales de 1km x 1km por lago ---")
    print(tabla.to_string(index=False))

    return resumen, tabla


# ============================================================
# 6.2 VISUALIZACION DE LOS BLOQUES
# ============================================================

def visualizar_bloques(df, nombre_lago, carpeta_salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(carpeta_salida, exist_ok=True)

    sub = df[df["lago"] == nombre_lago]

    fig, ax = plt.subplots(figsize=(7, 6))
    codigos_bloque = sub["bloque_id"].astype("category").cat.codes
    disp = ax.scatter(
        sub["x_utm"], sub["y_utm"],
        c=codigos_bloque, cmap="tab20", s=2,
    )
    ax.set_title(f"Bloques espaciales (1km x 1km) - {nombre_lago}")
    ax.set_xlabel("X UTM 15N (m)")
    ax.set_ylabel("Y UTM 15N (m)")
    ax.set_aspect("equal")
    fig.tight_layout()

    ruta = os.path.join(carpeta_salida, f"bloques_{nombre_lago.lower()}.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{nombre_lago}] Mapa de bloques guardado -> {ruta}")


# ============================================================
# 6.3 / 6.4 VALIDACION CRUZADA ESPACIAL (GroupKFold por bloque)
# ============================================================
# Se comparan dos esquemas de validacion cruzada con 5 folds sobre los
# MISMOS modelos e hiperparametros (los seleccionados en el Ejercicio
# 4.3), para que la unica diferencia entre ambos sea como se agrupan
# las observaciones en cada fold:
#   - aleatoria: StratifiedKFold, cada fold mezcla observaciones sin
#     considerar su cercania geografica (observaciones del mismo
#     bloque/pixel vecino pueden quedar repartidas entre train y test).
#   - espacial: GroupKFold usando bloque_id como grupo, de modo que
#     todas las observaciones de un mismo bloque de 1km x 1km quedan
#     siempre en el mismo lado (entrenamiento o prueba).

from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PREDICTORES = [
    "ndvi", "ndwi", "B02", "B03", "B08", "B11",
    "ratio_verde_azul", "brillo_superficial", "indice_turbidez_verde_swir",
]
COLUMNA_RESPUESTA = "alta_cianobacteria"
N_SPLITS_CV = 5
METRICAS_CV = ["f1", "recall", "precision", "roc_auc"]


def construir_modelos_para_cv(y):
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos

    return {
        "logistic_regression": LogisticRegression(C=100, max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
    }


def validacion_cruzada(df):
    X = df[PREDICTORES]
    y = df[COLUMNA_RESPUESTA]
    grupos = df["bloque_id"]

    modelos = construir_modelos_para_cv(y)

    cv_aleatoria = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
    cv_espacial = GroupKFold(n_splits=N_SPLITS_CV)

    filas = []
    for nombre, modelo in modelos.items():
        print(f"\nValidacion cruzada aleatoria - {nombre}...")
        res_aleatoria = cross_validate(modelo, X, y, cv=cv_aleatoria, scoring=METRICAS_CV, n_jobs=1)

        print(f"Validacion cruzada espacial (GroupKFold por bloque) - {nombre}...")
        res_espacial = cross_validate(modelo, X, y, cv=cv_espacial, groups=grupos, scoring=METRICAS_CV, n_jobs=1)

        for metrica in METRICAS_CV:
            filas.append({
                "modelo": nombre,
                "metrica": metrica,
                "validacion": "aleatoria",
                "media": res_aleatoria[f"test_{metrica}"].mean(),
                "std": res_aleatoria[f"test_{metrica}"].std(),
            })
            filas.append({
                "modelo": nombre,
                "metrica": metrica,
                "validacion": "espacial",
                "media": res_espacial[f"test_{metrica}"].mean(),
                "std": res_espacial[f"test_{metrica}"].std(),
            })

    return pd.DataFrame(filas)


# ============================================================
# 6.5 / 6.6 COMPARACION ALEATORIA VS ESPACIAL
# ============================================================

def comparar_validaciones(tabla_cv):
    pivote = tabla_cv.pivot_table(index=["modelo", "metrica"], columns="validacion", values="media").reset_index()
    pivote["diferencia_espacial_menos_aleatoria"] = pivote["espacial"] - pivote["aleatoria"]

    print("\n--- Comparacion validacion aleatoria vs espacial (media entre folds) ---")
    print(pivote.to_string(index=False))

    print(
        "\nInterpretacion esperada: la validacion espacial (GroupKFold por "
        "bloque de 1km) tiende a dar metricas iguales o mas bajas que la "
        "validacion aleatoria, porque en la aleatoria los pixeles vecinos "
        "de un mismo pixel de prueba (con valores espectrales casi "
        "identicos por autocorrelacion espacial) pueden quedar en el "
        "conjunto de entrenamiento, lo que infla artificialmente el "
        "desempeno reportado. La validacion espacial obliga al modelo a "
        "predecir sobre bloques geograficos nunca vistos, lo cual es una "
        "estimacion mas realista de su capacidad de generalizar a zonas "
        "nuevas del lago (que es, en la practica, para lo que se usaria "
        "un modelo de monitoreo)."
    )

    return pivote


if __name__ == "__main__":
    os.makedirs(CARPETA_TABLAS, exist_ok=True)
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)

    df = cargar_datos()
    df = reproyectar(df)
    df = asignar_bloques(df)

    resumen_bloques, tabla_resumen = evaluar_tamano_bloques(df)
    tabla_resumen.to_csv(os.path.join(CARPETA_TABLAS, "resumen_bloques_espaciales.csv"), index=False)
    resumen_bloques.to_csv(os.path.join(CARPETA_TABLAS, "observaciones_por_bloque.csv"), index=False)

    for lago in df["lago"].unique():
        visualizar_bloques(df, lago, CARPETA_FIGURAS)

    ruta_con_bloques = "parte2/resultados/dataset_con_bloques.csv"
    df.to_csv(ruta_con_bloques, index=False)
    print(f"\nDataset con bloques espaciales guardado -> {ruta_con_bloques}")

    tabla_cv = validacion_cruzada(df)
    tabla_cv.to_csv(os.path.join(CARPETA_TABLAS, "cv_aleatoria_vs_espacial.csv"), index=False)

    pivote_comparacion = comparar_validaciones(tabla_cv)
    pivote_comparacion.to_csv(os.path.join(CARPETA_TABLAS, "comparacion_aleatoria_vs_espacial.csv"), index=False)
