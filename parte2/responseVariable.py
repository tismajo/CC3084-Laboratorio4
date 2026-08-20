"""
Parte 2 - Ejercicio 2: Construccion de la variable respuesta.

Convierte el indice de cianobacteria (cyano_index, expresado como
clorofila-a estimada en microgramos/litro via el modelo NDCI de
Mishra & Mishra, 2012) en una variable binaria de alta/baja presencia,
usando un punto de corte basado en bibliografia cientifica.

Bibliografia consultada para el punto de corte:

  - World Health Organization (2003). Guidelines for Safe Recreational
    Water Environments. Volume 1: Coastal and Fresh Waters. Chapter 8:
    Algae and cyanobacteria in fresh water. Geneva: WHO.
    Define niveles de alerta para cianobacterias en aguas recreativas
    en funcion de la clorofila-a asociada a cianobacterias:
      * Nivel de Alerta 1 (vigilancia): ~10 ug/L de clorofila-a
        (o ~20,000 celulas/mL) -> probabilidad moderada de efectos
        adversos a la salud (irritacion de piel/mucosas).
      * Nivel de Alerta 2: ~50 ug/L de clorofila-a (o presencia de
        acumulaciones visibles/"scums") -> probabilidad alta de
        efectos adversos agudos a la salud.

  - Mishra, S., & Mishra, D. R. (2012). Normalized difference
    chlorophyll index: A novel model for remote estimation of
    chlorophyll-a concentration in turbid productive waters.
    Remote Sensing of Environment, 117, 394-406.
    Fuente del modelo NDCI usado para estimar clorofila-a (cyano_index)
    a partir de las bandas B04 (rojo) y B05 (borde rojo) de Sentinel-2.

Criterio adoptado: se usa el Nivel de Alerta 1 de la OMS (10 ug/L) como
punto de corte para la variable binaria, ya que corresponde al umbral
minimo en el que la literatura ambiental ya reporta riesgo sanitario
(vigilancia activa), en vez de esperar al escenario extremo de
floracion visible (Nivel 2, 50 ug/L). Esto es mas util para un modelo
de alerta temprana: 1 = alta presencia (riesgo sanitario, cyano_index
>= 10 ug/L), 0 = ausencia o baja presencia (< 10 ug/L).
"""

import os
import pandas as pd

RUTA_DATASET = "parte2/resultados/dataset_pixeles.csv"
CARPETA_TABLAS = "parte2/resultados/tablas"

UMBRAL_OMS_ALERTA1_UGL = 10.0


def cargar_dataset():
    return pd.read_csv(RUTA_DATASET, parse_dates=["fecha"])


# ============================================================
# 2.1 / 2.2 VARIABLE RESPUESTA BINARIA
# ============================================================

def construir_variable_respuesta(df, umbral=UMBRAL_OMS_ALERTA1_UGL):
    df = df.copy()
    df["alta_cianobacteria"] = (df["cyano_index"] >= umbral).astype(int)
    return df


# ============================================================
# 2.3 DISTRIBUCION GLOBAL DE LA VARIABLE RESPUESTA
# ============================================================

def distribucion_global(df):
    conteo = df["alta_cianobacteria"].value_counts().sort_index()
    porcentaje = (df["alta_cianobacteria"].value_counts(normalize=True) * 100).sort_index()

    resumen = pd.DataFrame({
        "clase": conteo.index,
        "n_observaciones": conteo.values,
        "porcentaje": porcentaje.values.round(3),
    })

    print("\n--- Distribucion global de la variable respuesta ---")
    print(resumen.to_string(index=False))

    return resumen


if __name__ == "__main__":
    os.makedirs(CARPETA_TABLAS, exist_ok=True)

    df = cargar_dataset()
    df = construir_variable_respuesta(df)

    ruta_con_respuesta = "parte2/resultados/dataset_con_respuesta.csv"
    df.to_csv(ruta_con_respuesta, index=False)
    print(f"Dataset con variable respuesta guardado -> {ruta_con_respuesta}")

    resumen_global = distribucion_global(df)
    resumen_global.to_csv(os.path.join(CARPETA_TABLAS, "distribucion_respuesta_global.csv"), index=False)
