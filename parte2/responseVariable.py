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


# ============================================================
# 2.3 DISTRIBUCION POR LAGO Y POR FECHA
# ============================================================

def distribucion_por_lago(df):
    tabla = (
        df.groupby("lago")["alta_cianobacteria"]
        .agg(n_observaciones="count", n_positivos="sum", tasa_positivos="mean")
        .reset_index()
    )
    tabla["tasa_positivos"] = (tabla["tasa_positivos"] * 100).round(3)

    print("\n--- Distribucion de la variable respuesta por lago ---")
    print(tabla.to_string(index=False))

    return tabla


def distribucion_por_fecha(df):
    tabla = (
        df.groupby(["lago", "fecha"])["alta_cianobacteria"]
        .agg(n_observaciones="count", n_positivos="sum", tasa_positivos="mean")
        .reset_index()
    )
    tabla["tasa_positivos"] = (tabla["tasa_positivos"] * 100).round(3)

    print("\n--- Distribucion de la variable respuesta por fecha ---")
    print(tabla.to_string(index=False))

    return tabla


# ============================================================
# 2.4 DESBALANCE DE CLASES
# ============================================================

def analizar_desbalance(resumen_global):
    fila_0 = resumen_global[resumen_global["clase"] == 0]
    fila_1 = resumen_global[resumen_global["clase"] == 1]

    n0 = int(fila_0["n_observaciones"].iloc[0]) if not fila_0.empty else 0
    n1 = int(fila_1["n_observaciones"].iloc[0]) if not fila_1.empty else 0

    razon = (n0 / n1) if n1 > 0 else float("inf")

    print("\n--- Desbalance de clases ---")
    print(f"Clase 0 (ausencia/baja): {n0} observaciones")
    print(f"Clase 1 (alta presencia): {n1} observaciones")
    print(f"Razon clase mayoritaria / minoritaria: {razon:.2f} : 1")
    print(
        "\nConsecuencias esperadas: con este nivel de desbalance, un modelo "
        "entrenado sin ajustes puede maximizar accuracy prediciendo "
        "casi siempre la clase mayoritaria (0), logrando accuracy alto pero "
        "recall muy bajo para la clase de interes (1 = alta cianobacteria), "
        "que es precisamente la clase que mas importa detectar desde el "
        "punto de vista de salud publica. Esto exige usar metricas "
        "sensibles al desbalance (recall, F1, ROC-AUC, PR-AUC) en vez de "
        "accuracy, y considerar tecnicas de balanceo (class_weight, "
        "sobremuestreo/submuestreo) al entrenar los modelos en el "
        "Ejercicio 4."
    )

    return {"n_clase_0": n0, "n_clase_1": n1, "razon_desbalance": razon}


# ============================================================
# 2.5 VARIABLES QUE NO PUEDEN USARSE COMO PREDICTORAS
# ============================================================

VARIABLES_EXCLUIDAS_POR_FUGA = {
    "cyano_index": (
        "Es la variable continua a partir de la cual se construye "
        "directamente la variable respuesta (alta_cianobacteria)."
    ),
    "B04": (
        "Banda roja usada directamente en el calculo de NDCI "
        "(NDCI = (B05-B04)/(B05+B04)), que alimenta la formula cubica "
        "con la que se calcula cyano_index. Usarla como predictor "
        "permitiria al modelo reconstruir casi exactamente la etiqueta."
    ),
    "B05": (
        "Banda de borde rojo (red edge) usada directamente en el "
        "calculo de NDCI, con el mismo problema de fuga que B04."
    ),
}

NOTA_NDVI = (
    "NDVI se calcula con B04 y B08 (misma B04 que interviene en NDCI), "
    "pero NDVI es una transformacion distinta a NDCI y no reproduce la "
    "relacion cubica NDCI->clorofila-a con la que se define la etiqueta; "
    "se mantiene como predictor valido, documentando este riesgo menor "
    "de colinealidad indirecta con la respuesta."
)


def imprimir_variables_excluidas():
    print("\n--- Variables excluidas como predictoras (fuga de informacion) ---")
    for variable, motivo in VARIABLES_EXCLUIDAS_POR_FUGA.items():
        print(f"- {variable}: {motivo}")
    print(f"\nNota sobre NDVI: {NOTA_NDVI}")


if __name__ == "__main__":
    os.makedirs(CARPETA_TABLAS, exist_ok=True)

    df = cargar_dataset()
    df = construir_variable_respuesta(df)

    ruta_con_respuesta = "parte2/resultados/dataset_con_respuesta.csv"
    df.to_csv(ruta_con_respuesta, index=False)
    print(f"Dataset con variable respuesta guardado -> {ruta_con_respuesta}")

    resumen_global = distribucion_global(df)
    resumen_global.to_csv(os.path.join(CARPETA_TABLAS, "distribucion_respuesta_global.csv"), index=False)

    tabla_lago = distribucion_por_lago(df)
    tabla_lago.to_csv(os.path.join(CARPETA_TABLAS, "distribucion_respuesta_por_lago.csv"), index=False)

    tabla_fecha = distribucion_por_fecha(df)
    tabla_fecha.to_csv(os.path.join(CARPETA_TABLAS, "distribucion_respuesta_por_fecha.csv"), index=False)

    analizar_desbalance(resumen_global)

    imprimir_variables_excluidas()
