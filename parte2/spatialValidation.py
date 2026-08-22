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
