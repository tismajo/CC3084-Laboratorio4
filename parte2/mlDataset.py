"""
Parte 2 - Ejercicio 1: Preparacion de los datos para Machine Learning.

Construye un conjunto de datos a nivel de pixel (coordenadas, fecha, lago,
bandas espectrales, NDVI, NDWI e indice de cianobacteria) a partir de la
API de Sentinel Hub, usando las mismas fechas y bounding boxes definidos
en la Parte I (ver indices.py).

Nota tecnica: en esta maquina rasterio/GDAL esta bloqueado por una
politica de Control de aplicaciones de Windows (WDAC), por lo que este
script evita rasterio por completo. En vez de descargar .tif y leerlos
con rasterio (como en indices.py), se piden los arreglos directamente a
la API de Sentinel Hub (numpy en memoria) y las coordenadas de cada pixel
se calculan manualmente a partir del bounding box, que es equivalente a
la transformacion afin que usaria rasterio para georreferenciar.
"""

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox,
    bbox_to_dimensions,
)

load_dotenv()
config = SHConfig()
config.sh_client_id = os.environ.get("SH_CLIENT_ID")
config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")

if not config.sh_client_id or not config.sh_client_secret:
    raise RuntimeError(
        "Faltan SH_CLIENT_ID o SH_CLIENT_SECRET. Verifica tu archivo .env."
    )

config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

SENTINEL2_L1C_CDSE = DataCollection.SENTINEL2_L1C.define_from(
    "SENTINEL2_L1C_CDSE_ML",
    service_url=config.sh_base_url
)


# ============================================================
# LAGOS Y FECHAS (identicas a la Parte I, ver indices.py)
# ============================================================

lago_atitlan = {
    "west": -91.326256,
    "east": -91.07151,
    "south": 14.5948,
    "north": 14.750979,
}

lago_amatitlan = {
    "west": -90.638065,
    "east": -90.512924,
    "south": 14.412347,
    "north": 14.493799,
}

fechas_atitlan = [
    "2025-01-18", "2025-04-13", "2025-05-13", "2025-07-17",
    "2025-11-21", "2025-12-29", "2026-02-12", "2026-03-24",
    "2026-04-13", "2026-04-28", "2026-07-22",
]

fechas_amatitlan = [
    "2025-01-28", "2025-04-15", "2025-04-28", "2025-11-24",
    "2026-01-08", "2026-02-02", "2026-02-07", "2026-03-29",
    "2026-04-13", "2026-04-28", "2026-06-19",
]

# Resolucion espacial usada para el dataset de ML.
# Se eligio 50 m (en vez de los 20 m usados para los mapas de la Parte I)
# para mantener manejable el volumen de datos descargados via API y el
# tiempo de ejecucion, conservando suficiente detalle para bloques
# espaciales de ~1 km x 1 km (Ejercicio 6): cada bloque contendria
# aproximadamente 20 x 20 pixeles a esta resolucion.
RESOLUCION_M = 50

CARPETA_CACHE = "parte2/data_cache"
CARPETA_RESULTADOS = "parte2/resultados"
CARPETA_TABLAS = os.path.join(CARPETA_RESULTADOS, "tablas")


def crear_carpetas():
    for carpeta in [CARPETA_CACHE, CARPETA_RESULTADOS, CARPETA_TABLAS]:
        os.makedirs(carpeta, exist_ok=True)


# ============================================================
# EVALSCRIPT COMBINADO
# ============================================================
# Combina en una sola peticion:
#   - deteccion de agua (mismo criterio WBI usado en indices.py, Parte I)
#   - NDVI y NDWI (misma formula usada en indices.py, Parte I)
#   - indice de cianobacteria via NDCI (mismo modelo usado en indices.py,
#     Parte I; NDCI y su regresion cubica de clorofila-a son de
#     Mishra & Mishra, 2012, Remote Sensing of Environment)
#   - bandas espectrales crudas de interes como posibles predictores
#
# Bandas de salida (en este orden): ndvi, ndwi, cyano, water,
#                                    B02, B03, B04, B05, B08, B11
EVALSCRIPT_ML = """
//VERSION=3
function setup() {
  return {
    input: ["B02","B03","B04","B05","B08","B11","B12"],
    output: { bands: 10, sampleType: "FLOAT32" }
  };
}

function wbi(r,g,b,nir,swir1,swir2) {
  let ws = 0;
  try {
    var ndvi=(nir-r)/(nir+r), mndwi=(g-swir1)/(g+swir1), ndwi=(g-nir)/(g+nir),
        ndwi_leaves=(nir-swir1)/(nir+swir1),
        aweish=b+2.5*g-1.5*(nir+swir1)-0.25*swir2,
        aweinsh=4*(g-swir1)-(0.25*nir+2.75*swir1);
    var dbsi=((swir1-g)/(swir1+g))-ndvi;
    if (mndwi>0.42||ndwi>0.4||aweinsh>0.1879||aweish>0.1112||ndvi<-0.2||ndwi_leaves>1) { ws=1; }
    if (ws==1 && ((aweinsh<=-0.03)||(dbsi>0))) { ws=0; }
  } catch(err) { ws=0; }
  return ws;
}

function evaluatePixel(s) {
  let water = wbi(s.B04, s.B03, s.B02, s.B08, s.B11, s.B12);

  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
  let ndwi = (s.B03 - s.B08) / (s.B03 + s.B08);

  function NDCI(a,b) { return (b-a)/(b+a); }
  let NDCIv = NDCI(s.B04, s.B05);
  let chl = 826.57*Math.pow(NDCIv,3) - 176.43*Math.pow(NDCIv,2) + 19*NDCIv + 4.071;

  if (water == 0) {
    return [NaN, NaN, NaN, 0, s.B02, s.B03, s.B04, s.B05, s.B08, s.B11];
  }
  return [ndvi, ndwi, chl, 1, s.B02, s.B03, s.B04, s.B05, s.B08, s.B11];
}
"""

NOMBRES_BANDAS_SALIDA = [
    "ndvi", "ndwi", "cyano_index", "is_water",
    "B02", "B03", "B04", "B05", "B08", "B11",
]


def descargar_pixeles(nombre_lago, bbox_dict, fecha, resolucion=RESOLUCION_M):
    """
    Descarga (o carga desde cache local) el arreglo combinado de
    indices/bandas para una fecha y un lago. Devuelve un arreglo
    numpy (alto, ancho, 10).
    """

    carpeta_lago = os.path.join(CARPETA_CACHE, nombre_lago.lower())
    os.makedirs(carpeta_lago, exist_ok=True)
    ruta_cache = os.path.join(carpeta_lago, f"{fecha}.npz")

    if os.path.exists(ruta_cache):
        return np.load(ruta_cache)["datos"]

    bbox = BBox(
        bbox=[bbox_dict["west"], bbox_dict["south"], bbox_dict["east"], bbox_dict["north"]],
        crs=CRS.WGS84,
    )
    ancho, alto = bbox_to_dimensions(bbox, resolution=resolucion)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_ML,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=SENTINEL2_L1C_CDSE,
                time_interval=(fecha, fecha),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(ancho, alto),
        config=config,
    )

    resultado = request.get_data(save_data=False)[0]
    resultado = np.asarray(resultado, dtype="float32")

    np.savez_compressed(ruta_cache, datos=resultado)
    return resultado


def construir_filas(nombre_lago, fecha, datos, bbox_dict):
    """
    Aplana el arreglo (alto, ancho, 10) a un DataFrame de observaciones
    validas (una fila por pixel de agua), calculando la coordenada
    (lon, lat) del centro de cada pixel a partir del bounding box.

    Esta interpolacion lineal es equivalente a la transformacion afin
    que usa rasterio (from_bounds) para georreferenciar un raster.
    """

    alto, ancho, _ = datos.shape

    col = np.arange(ancho)
    fila = np.arange(alto)

    lon = bbox_dict["west"] + (col + 0.5) / ancho * (bbox_dict["east"] - bbox_dict["west"])
    lat = bbox_dict["north"] - (fila + 0.5) / alto * (bbox_dict["north"] - bbox_dict["south"])

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    df = pd.DataFrame({
        "lon": lon_grid.ravel(),
        "lat": lat_grid.ravel(),
    })

    for i, nombre in enumerate(NOMBRES_BANDAS_SALIDA):
        df[nombre] = datos[:, :, i].ravel()

    df["lago"] = nombre_lago
    df["fecha"] = fecha

    return df


def construir_dataset(nombre_lago, bbox_dict, fechas):
    piezas = []
    for fecha in fechas:
        try:
            datos = descargar_pixeles(nombre_lago, bbox_dict, fecha)
        except Exception as e:
            print(f"[{nombre_lago}] ERROR descargando {fecha}: {e}")
            continue

        df_fecha = construir_filas(nombre_lago, fecha, datos, bbox_dict)
        piezas.append(df_fecha)
        print(f"[{nombre_lago}] {fecha}: {len(df_fecha)} pixeles totales descargados.")

    if not piezas:
        return pd.DataFrame()

    return pd.concat(piezas, ignore_index=True)


# ============================================================
# 1.3 LIMPIEZA: eliminar pixeles fuera del lago / NoData / invalidos
# ============================================================

def limpiar_dataset(df):
    """
    Elimina:
      - pixeles que la mascara de agua (is_water) marca como no-agua
        (equivalente a "fuera de los limites del lago")
      - pixeles con NoData / valores no finitos en ndvi, ndwi o
        cyano_index (nubes, bordes sin cobertura, errores de division)
    """

    n_inicial = len(df)

    df = df[df["is_water"] == 1].copy()
    n_agua = len(df)

    columnas_criticas = ["ndvi", "ndwi", "cyano_index"]
    df = df[np.isfinite(df[columnas_criticas]).all(axis=1)].copy()
    n_final = len(df)

    # Filtro adicional de sanidad fisica: ndvi y ndwi estan definidos
    # matematicamente en [-1, 1]; valores fuera de rango indican pixeles
    # saturados o con errores numericos y se descartan.
    df = df[(df["ndvi"].between(-1, 1)) & (df["ndwi"].between(-1, 1))].copy()
    n_rango = len(df)

    df = df.drop(columns=["is_water"])

    print("\n--- Limpieza del dataset ---")
    print(f"Filas iniciales (todos los pixeles del bbox): {n_inicial}")
    print(f"Filas tras filtrar fuera del lago (mascara de agua): {n_agua}")
    print(f"Filas tras eliminar NoData/valores no finitos:       {n_final}")
    print(f"Filas tras filtrar ndvi/ndwi fuera de [-1,1]:        {n_rango}")

    return df.reset_index(drop=True)


# ============================================================
# 1.4 RESUMEN DEL DATASET
# ============================================================

def resumen_dataset(df):
    print("\n" + "=" * 60)
    print("RESUMEN DEL CONJUNTO DE DATOS PARA MACHINE LEARNING")
    print("=" * 60)

    print(f"\nNumero total de observaciones: {len(df)}")

    print("\nObservaciones por lago:")
    print(df["lago"].value_counts().to_string())

    print("\nObservaciones por lago y fecha:")
    print(df.groupby(["lago", "fecha"]).size().to_string())

    print("\nTipo de cada variable:")
    print(df.dtypes.to_string())

    print("\nPorcentaje de valores faltantes por variable:")
    faltantes = (df.isna().mean() * 100).round(3)
    print(faltantes.to_string())

    return faltantes


# ============================================================
# 1.5 ANALISIS EXPLORATORIO
# ============================================================

CARPETA_FIGURAS_EDA = os.path.join(CARPETA_RESULTADOS, "figuras", "eda")

VARIABLES_NUMERICAS_EDA = [
    "ndvi", "ndwi", "cyano_index", "B02", "B03", "B04", "B05", "B08", "B11"
]


def analisis_exploratorio(df):
    """
    Genera estadisticas descriptivas y visualizaciones de las variables
    que se usaran para construir los modelos (NDVI, NDWI, cyano_index y
    bandas espectrales), tanto de forma global como por lago.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(CARPETA_FIGURAS_EDA, exist_ok=True)

    # --- Estadisticas descriptivas ---
    descriptivas = df[VARIABLES_NUMERICAS_EDA].describe().T
    descriptivas.to_csv(os.path.join(CARPETA_TABLAS, "estadisticas_descriptivas_eda.csv"))
    print("\n--- Estadisticas descriptivas (todas las observaciones) ---")
    print(descriptivas.to_string())

    descriptivas_por_lago = df.groupby("lago")[VARIABLES_NUMERICAS_EDA].describe()
    descriptivas_por_lago.to_csv(os.path.join(CARPETA_TABLAS, "estadisticas_descriptivas_por_lago.csv"))

    # --- Histogramas de ndvi, ndwi y cyano_index ---
    # cyano_index se recorta al percentil 1-99 solo para la visualizacion
    # (no para los datos en si), ya que la regresion cubica NDCI produce
    # unos pocos valores extremos no acotados que saturan la escala del
    # histograma (ver Decision 8 en imprimir_decisiones()).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["ndvi", "ndwi", "cyano_index"]):
        valores = df[col].dropna()
        if col == "cyano_index":
            p1, p99 = valores.quantile([0.01, 0.99])
            valores = valores.clip(p1, p99)
        ax.hist(valores, bins=60, color="#2b7a78")
        ax.set_title(f"Distribucion de {col}" + (" (recortado P1-P99)" if col == "cyano_index" else ""))
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    ruta_hist = os.path.join(CARPETA_FIGURAS_EDA, "histogramas_indices.png")
    fig.savefig(ruta_hist, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogramas guardados -> {ruta_hist}")

    # --- Boxplot de cyano_index por lago (escala log para visibilidad) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    datos_boxplot = [
        df.loc[df["lago"] == lago, "cyano_index"].clip(lower=0.01)
        for lago in df["lago"].unique()
    ]
    ax.boxplot(datos_boxplot, tick_labels=df["lago"].unique(), showfliers=False)
    ax.set_yscale("log")
    ax.set_title("Distribucion de cyano_index por lago (escala log, sin outliers)")
    ax.set_ylabel("cyano_index (clorofila-a estimada, ug/L)")
    fig.tight_layout()
    ruta_box = os.path.join(CARPETA_FIGURAS_EDA, "boxplot_cyano_por_lago.png")
    fig.savefig(ruta_box, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Boxplot por lago guardado -> {ruta_box}")

    # --- Matriz de correlacion entre variables numericas ---
    corr = df[VARIABLES_NUMERICAS_EDA].corr()
    corr.to_csv(os.path.join(CARPETA_TABLAS, "correlacion_variables_eda.csv"))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Correlacion")
    ax.set_title("Correlacion entre variables espectrales e indices")
    fig.tight_layout()
    ruta_corr = os.path.join(CARPETA_FIGURAS_EDA, "correlacion_variables.png")
    fig.savefig(ruta_corr, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Matriz de correlacion guardada -> {ruta_corr}")

    return descriptivas, corr


# ============================================================
# 1.6 DECISIONES DE PREPARACION Y LIMPIEZA
# ============================================================

DECISIONES_PREPARACION = """
--- Decisiones tomadas durante la preparacion y limpieza de los datos ---

1. Fuente de los datos: se mantienen exactamente los mismos lagos,
   bounding boxes y fechas usados en la Parte I (ver indices.py), para
   que el analisis de la Parte 2 sea directamente comparable con el de
   la Parte 1.

2. Extraccion sin rasterio: en el equipo usado para este avance,
   rasterio/GDAL esta bloqueado por una politica de Control de
   aplicaciones de Windows (WDAC) al nivel de sistema operativo (no es
   un problema de dependencias de Python). Por eso, en vez de descargar
   .tif y leerlos con rasterio como en indices.py, se piden los
   arreglos directamente a la API de Sentinel Hub (numpy en memoria) y
   las coordenadas de cada pixel se calculan por interpolacion lineal
   a partir del bounding box, lo cual es equivalente a la
   transformacion afin que usaria rasterio para georreferenciar.

3. Resolucion espacial (50 m): se redujo de 20 m (usados para los
   mapas de la Parte I) a 50 m para mantener manejable el volumen de
   datos descargados via API dentro del tiempo disponible, conservando
   suficiente densidad de pixeles por bloque de ~1 km x 1 km
   (Ejercicio 6: ~20x20 pixeles por bloque a esta resolucion).

4. Mascara de agua: se usa el mismo criterio WBI (indice booleano de
   agua) que la Parte I para descartar pixeles fuera del lago (tierra,
   vegetacion ribereña, nubes sobre tierra), en vez de recortar por un
   poligono fijo, porque el area de agua cambia ligeramente entre
   fechas (nivel del lago, nubes).

5. Valores no finitos: se descartan filas con NaN/Inf en ndvi, ndwi o
   cyano_index. Estos casos corresponden a division por cero (por
   ejemplo B08+B04=0) o a pixeles marcados como no-agua por la
   evalscript (que devuelve NaN explicitamente en esos casos).

6. Rango fisico valido: se descartan pixeles con ndvi o ndwi fuera de
   [-1, 1], el rango matematicamente valido de estos indices; en la
   practica esto elimino un numero minimo de filas (ver conteo en la
   funcion limpiar_dataset), confirmando que la mascara de agua ya
   habia retirado la mayoria de los pixeles problematicos.

7. cyano_index como clorofila-a estimada: se conserva la formula NDCI
   original de Mishra & Mishra (2012) usada en la Parte I, sin
   recalibrar, para mantener consistencia con los resultados ya
   obtenidos (resultados/*.csv de la Parte I).

8. Valores extremos de cyano_index: al trabajar a nivel de pixel (en
   vez de promedios por fecha como en la Parte I) se observan valores
   minimos/maximos fuera del rango tipico reportado en Parte I
   (ver estadisticas_descriptivas_eda.csv). Esto ocurre porque la
   regresion cubica NDCI->clorofila-a no esta acotada y unos pocos
   pixeles de agua con NDCI cercano a los extremos matematicos
   producen valores extremos. Se documenta como limitacion conocida
   (ver Ejercicio 10.2) en vez de recortarse arbitrariamente, para no
   introducir un sesgo adicional no justificado en la variable
   respuesta antes del Ejercicio 2.
"""


def imprimir_decisiones():
    print(DECISIONES_PREPARACION)


if __name__ == "__main__":
    crear_carpetas()

    df_atitlan = construir_dataset("Atitlan", lago_atitlan, fechas_atitlan)
    df_amatitlan = construir_dataset("Amatitlan", lago_amatitlan, fechas_amatitlan)

    df_completo = pd.concat([df_atitlan, df_amatitlan], ignore_index=True)

    df_limpio = limpiar_dataset(df_completo)

    faltantes = resumen_dataset(df_limpio)

    ruta_dataset = os.path.join(CARPETA_RESULTADOS, "dataset_pixeles.csv")
    df_limpio.to_csv(ruta_dataset, index=False)
    print(f"\nDataset guardado -> {ruta_dataset}")

    ruta_resumen = os.path.join(CARPETA_TABLAS, "resumen_dataset_ml.csv")
    resumen_por_lago_fecha = df_limpio.groupby(["lago", "fecha"]).size().reset_index(name="n_observaciones")
    resumen_por_lago_fecha.to_csv(ruta_resumen, index=False)
    print(f"Resumen por lago/fecha guardado -> {ruta_resumen}")

    ruta_faltantes = os.path.join(CARPETA_TABLAS, "valores_faltantes_ml.csv")
    faltantes.reset_index().rename(columns={"index": "variable", 0: "pct_faltante"}).to_csv(
        ruta_faltantes, index=False
    )
    print(f"Tabla de valores faltantes guardada -> {ruta_faltantes}")

    analisis_exploratorio(df_limpio)
    imprimir_decisiones()
