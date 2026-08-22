"""
Parte 2 - Ejercicio 9: Generacion de mapas predictivos.

Usa el mejor modelo (XGBoost, seleccionado en el Ejercicio 5.3 por
criterio ambiental) para calcular la probabilidad de alta presencia de
cianobacteria en cada observacion, y reconstruye espacialmente esas
probabilidades en un mapa por lago.
"""

import os
import joblib
import numpy as np
import pandas as pd

RUTA_DATOS = "parte2/resultados/dataset_features.csv"
CARPETA_MODELOS = "parte2/resultados/modelos"
CARPETA_TABLAS = "parte2/resultados/tablas"
CARPETA_FIGURAS = "parte2/resultados/figuras/mapas_predictivos"

PREDICTORES = [
    "ndvi", "ndwi", "B02", "B03", "B08", "B11",
    "ratio_verde_azul", "brillo_superficial", "indice_turbidez_verde_swir",
]
COLUMNA_RESPUESTA = "alta_cianobacteria"
MODELO_SELECCIONADO = "xgboost"

# Fecha representativa por lago para el mapa (la fecha con mayor tasa
# de positivos observada en el Ejercicio 2.3, para que el mapa muestre
# un evento de floracion real en vez de un dia sin actividad).
FECHA_REPRESENTATIVA = {
    "Atitlan": "2026-07-22",
    "Amatitlan": "2026-06-19",
}

# Bins de probabilidad para la escala del mapa (Ejercicio 9.4).
BINS_PROBABILIDAD = [0, 0.1, 0.3, 0.6, 1.0]
ETIQUETAS_PROBABILIDAD = ["muy baja", "baja", "alta", "muy alta"]


def cargar_datos_y_modelo():
    df = pd.read_csv(RUTA_DATOS, parse_dates=["fecha"])
    modelo = joblib.load(os.path.join(CARPETA_MODELOS, f"{MODELO_SELECCIONADO}_final.joblib"))
    return df, modelo


# ============================================================
# 9.1 PROBABILIDAD DE ALTA PRESENCIA PARA CADA OBSERVACION
# ============================================================

def calcular_probabilidades(df, modelo):
    df = df.copy()
    df["probabilidad_alta_cianobacteria"] = modelo.predict_proba(df[PREDICTORES])[:, 1]
    df["prediccion"] = (df["probabilidad_alta_cianobacteria"] >= 0.5).astype(int)
    return df


# ============================================================
# 9.2 / 9.3 / 9.4 MAPA DE PROBABILIDAD POR LAGO
# ============================================================

def generar_mapa_probabilidad(df, nombre_lago, fecha, carpeta_salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    os.makedirs(carpeta_salida, exist_ok=True)

    sub = df[(df["lago"] == nombre_lago) & (df["fecha"] == fecha)]
    if sub.empty:
        print(f"[{nombre_lago}] No hay observaciones para {fecha}, se omite el mapa.")
        return

    cmap = ListedColormap(["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"])
    norm = BoundaryNorm(BINS_PROBABILIDAD, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ax.scatter(
        sub["lon"], sub["lat"],
        c=sub["probabilidad_alta_cianobacteria"],
        cmap=cmap, norm=norm, s=3,
    )
    ax.set_title(f"Probabilidad de alta presencia de cianobacteria\n{nombre_lago} - {fecha}")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    cbar = fig.colorbar(disp, ax=ax, ticks=[0.05, 0.2, 0.45, 0.8])
    cbar.ax.set_yticklabels(ETIQUETAS_PROBABILIDAD)
    cbar.set_label("Probabilidad de alta presencia")

    fig.tight_layout()
    ruta = os.path.join(carpeta_salida, f"mapa_probabilidad_{nombre_lago.lower()}_{fecha}.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{nombre_lago}] Mapa predictivo guardado -> {ruta}")


if __name__ == "__main__":
    os.makedirs(CARPETA_TABLAS, exist_ok=True)
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)

    df, modelo = cargar_datos_y_modelo()
    df = calcular_probabilidades(df, modelo)

    for nombre_lago, fecha in FECHA_REPRESENTATIVA.items():
        generar_mapa_probabilidad(df, nombre_lago, fecha, CARPETA_FIGURAS)

    ruta_probabilidades = os.path.join(CARPETA_TABLAS, "resumen_probabilidades_por_fecha.csv")
    resumen = df.groupby(["lago", "fecha"])["probabilidad_alta_cianobacteria"].agg(["mean", "median", "max"]).reset_index()
    resumen.to_csv(ruta_probabilidades, index=False)
    print(f"\nResumen de probabilidades por fecha guardado -> {ruta_probabilidades}")
