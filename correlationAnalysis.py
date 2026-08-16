import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from rasterio.warp import reproject, Resampling


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

CARPETA_TABLAS = "resultados/tablas"
CARPETA_FIGURAS = "resultados/figuras/correlaciones"
CARPETA_ATITLAN = os.path.join(CARPETA_FIGURAS, "atitlan")
CARPETA_AMATITLAN = os.path.join(CARPETA_FIGURAS, "amatitlan")

for carpeta in [CARPETA_TABLAS, CARPETA_FIGURAS, CARPETA_ATITLAN, CARPETA_AMATITLAN]:
    os.makedirs(carpeta, exist_ok=True)


def reprojectar_a_referencia(ruta_indice, referencia):
    """Reproyecta NDVI/NDWI a la misma grilla del raster de cianobacteria."""
    with rasterio.open(ruta_indice) as src:
        destino = np.full((referencia.height, referencia.width), np.nan, dtype="float32")

        reproject(
            source=rasterio.band(src, 1),
            destination=destino,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=referencia.transform,
            dst_crs=referencia.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    destino[~np.isfinite(destino)] = np.nan
    return destino


def correlacion_pearson(x, y):
    """Calcula correlacion de Pearson sin dependencias adicionales."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def muestrear(x, y, max_puntos=12000):
    """Reduce puntos para que las graficas de dispersion no sean demasiado pesadas."""
    if len(x) <= max_puntos:
        return x, y

    rng = np.random.default_rng(42)
    indices = rng.choice(len(x), size=max_puntos, replace=False)
    return x[indices], y[indices]


def analizar_fecha(nombre_lago, carpeta_lago, fecha, carpeta_salida):
    ruta_cyano = os.path.join(carpeta_lago, f"{fecha}_cyano.tif")
    ruta_ndvi = os.path.join(carpeta_lago, f"{fecha}_ndvi.tif")
    ruta_ndwi = os.path.join(carpeta_lago, f"{fecha}_ndwi.tif")

    faltantes = [ruta for ruta in [ruta_cyano, ruta_ndvi, ruta_ndwi] if not os.path.exists(ruta)]
    if faltantes:
        print(f"[{nombre_lago}] {fecha}: faltan archivos, se omite.")
        for ruta in faltantes:
            print(f"   - {ruta}")
        return None

    with rasterio.open(ruta_cyano) as ref:
        cyano = ref.read(1).astype("float32")
        ndvi = reprojectar_a_referencia(ruta_ndvi, ref)
        ndwi = reprojectar_a_referencia(ruta_ndwi, ref)

    cyano[~np.isfinite(cyano)] = np.nan

    mascara_ndvi = (
        np.isfinite(cyano)
        & np.isfinite(ndvi)
        & (ndvi >= -1.01)
        & (ndvi <= 1.01)
    )
    mascara_ndwi = (
        np.isfinite(cyano)
        & np.isfinite(ndwi)
        & (ndwi >= -1.01)
        & (ndwi <= 1.01)
    )

    cyano_ndvi = cyano[mascara_ndvi]
    valores_ndvi = ndvi[mascara_ndvi]
    cyano_ndwi = cyano[mascara_ndwi]
    valores_ndwi = ndwi[mascara_ndwi]

    corr_ndvi = correlacion_pearson(valores_ndvi, cyano_ndvi)
    corr_ndwi = correlacion_pearson(valores_ndwi, cyano_ndwi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_ndvi, y_ndvi = muestrear(valores_ndvi, cyano_ndvi)
    x_ndwi, y_ndwi = muestrear(valores_ndwi, cyano_ndwi)

    axes[0].scatter(x_ndvi, y_ndvi, s=7, alpha=0.25)
    axes[0].set_title(f"NDVI vs Cianobacteria\nr = {corr_ndvi:.3f}")
    axes[0].set_xlabel("NDVI")
    axes[0].set_ylabel("Clorofila-a estimada")

    axes[1].scatter(x_ndwi, y_ndwi, s=7, alpha=0.25)
    axes[1].set_title(f"NDWI vs Cianobacteria\nr = {corr_ndwi:.3f}")
    axes[1].set_xlabel("NDWI")
    axes[1].set_ylabel("Clorofila-a estimada")

    fig.suptitle(f"{nombre_lago} - {fecha}")
    fig.tight_layout()

    ruta_figura = os.path.join(carpeta_salida, f"{fecha}_correlaciones.png")
    fig.savefig(ruta_figura, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[{nombre_lago}] {fecha}: r(NDVI)={corr_ndvi:.3f}, r(NDWI)={corr_ndwi:.3f}")

    return {
        "lago": nombre_lago,
        "fecha": fecha,
        "correlacion_ndvi_cyano": corr_ndvi,
        "correlacion_ndwi_cyano": corr_ndwi,
        "pixeles_ndvi": int(len(valores_ndvi)),
        "pixeles_ndwi": int(len(valores_ndwi)),
    }


def procesar_lago(nombre_lago, carpeta_lago, fechas, carpeta_salida):
    registros = []
    for fecha in fechas:
        resultado = analizar_fecha(nombre_lago, carpeta_lago, fecha, carpeta_salida)
        if resultado is not None:
            registros.append(resultado)
    return pd.DataFrame(registros)


df_atitlan = procesar_lago("Atitlan", "data/atitlan", fechas_atitlan, CARPETA_ATITLAN)
df_amatitlan = procesar_lago("Amatitlan", "data/amatitlan", fechas_amatitlan, CARPETA_AMATITLAN)

df_correlaciones = pd.concat([df_atitlan, df_amatitlan], ignore_index=True)

if not df_correlaciones.empty:
    df_correlaciones["fecha"] = pd.to_datetime(df_correlaciones["fecha"])

    ruta_tabla = os.path.join(CARPETA_TABLAS, "correlaciones_indices_cianobacteria.csv")
    df_correlaciones.to_csv(ruta_tabla, index=False)

    resumen_lagos = (
        df_correlaciones.groupby("lago")
        .agg(
            fechas_analizadas=("fecha", "count"),
            correlacion_ndvi_media=("correlacion_ndvi_cyano", "mean"),
            correlacion_ndvi_mediana=("correlacion_ndvi_cyano", "median"),
            correlacion_ndwi_media=("correlacion_ndwi_cyano", "mean"),
            correlacion_ndwi_mediana=("correlacion_ndwi_cyano", "median"),
        )
        .reset_index()
    )

    ruta_resumen = os.path.join(CARPETA_TABLAS, "resumen_correlaciones_por_lago.csv")
    resumen_lagos.to_csv(ruta_resumen, index=False)

    for lago, df_lago in df_correlaciones.groupby("lago"):
        df_lago = df_lago.sort_values("fecha")
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            df_lago["fecha"],
            df_lago["correlacion_ndvi_cyano"],
            marker="o",
            label="NDVI vs cianobacteria"
        )
        ax.plot(
            df_lago["fecha"],
            df_lago["correlacion_ndwi_cyano"],
            marker="o",
            label="NDWI vs cianobacteria"
        )

        ax.axhline(0, linewidth=1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Evolucion de correlaciones - {lago}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Correlacion de Pearson (r)")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.autofmt_xdate()
        fig.tight_layout()

        ruta = os.path.join(CARPETA_FIGURAS, f"evolucion_correlaciones_{lago.lower()}.png")
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\nTabla guardada ->", ruta_tabla)
    print("Resumen guardado ->", ruta_resumen)
    print("\nResumen de correlaciones:")
    print(resumen_lagos.to_string(index=False))
else:
    print("No fue posible calcular correlaciones.")

print("\nAnalisis de correlacion finalizado")