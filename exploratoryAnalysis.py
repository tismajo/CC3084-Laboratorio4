"""
Laboratorio 4 - Ejercicio 8
Analisis exploratorio adicional de cianobacteria.

Genera:
    8.1 Porcentaje del lago con valores altos por fecha.
    8.2 Mapa de zonas persistentes de acumulacion.
    8.3 Boxplots y mapas de diferencia entre fechas.
    8.4 Resumen trimestral para explorar patrones temporales/estacionales.
    8.5 Tablas de apoyo para la interpretacion.

Nota:
    El umbral de "valor alto" es un umbral relativo por lago,
    estimado como el percentil 75 de los valores validos de todas
    las fechas disponibles. No es un limite sanitario.
"""

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
CARPETA_FIGURAS = "resultados/figuras/exploratorio"
CARPETA_RASTERS = "resultados/rasters/exploratorio"
CARPETA_ATITLAN = os.path.join(CARPETA_FIGURAS, "atitlan")
CARPETA_AMATITLAN = os.path.join(CARPETA_FIGURAS, "amatitlan")

for carpeta in [
    CARPETA_TABLAS,
    CARPETA_FIGURAS,
    CARPETA_RASTERS,
    CARPETA_ATITLAN,
    CARPETA_AMATITLAN,
]:
    os.makedirs(carpeta, exist_ok=True)


def ruta_cyano(carpeta_lago, fecha):
    return os.path.join(carpeta_lago, f"{fecha}_cyano.tif")


def obtener_muestra_para_umbral(carpeta_lago, fechas, max_por_fecha=100000):
    """Toma una muestra de pixeles validos para estimar un umbral relativo."""
    muestras = []
    rng = np.random.default_rng(42)

    for fecha in fechas:
        ruta = ruta_cyano(carpeta_lago, fecha)
        if not os.path.exists(ruta):
            continue

        with rasterio.open(ruta) as src:
            datos = src.read(1).astype("float32")

        valores = datos[np.isfinite(datos)]
        if valores.size == 0:
            continue

        if valores.size > max_por_fecha:
            indices = rng.choice(valores.size, size=max_por_fecha, replace=False)
            valores = valores[indices]

        muestras.append(valores)

    if not muestras:
        return np.array([], dtype="float32")

    return np.concatenate(muestras)


def reproyectar_array_a_referencia(ruta, referencia, resampling=Resampling.bilinear):
    """Lleva un raster a la misma grilla de un raster de referencia."""
    with rasterio.open(ruta) as src:
        destino = np.full(
            (referencia.height, referencia.width),
            np.nan,
            dtype="float32"
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=destino,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=referencia.transform,
            dst_crs=referencia.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    destino[~np.isfinite(destino)] = np.nan
    return destino


def calcular_extension_por_fecha(nombre_lago, carpeta_lago, fechas, umbral):
    """Calcula porcentaje de pixeles de agua con valores altos por fecha."""
    registros = []

    for fecha in fechas:
        ruta = ruta_cyano(carpeta_lago, fecha)
        if not os.path.exists(ruta):
            print(f"[{nombre_lago}] Falta {ruta}.")
            continue

        with rasterio.open(ruta) as src:
            datos = src.read(1).astype("float32")

        mascara = np.isfinite(datos)
        valores = datos[mascara]

        if valores.size == 0:
            continue

        porcentaje_alto = float(100 * np.sum(valores >= umbral) / valores.size)

        registros.append({
            "lago": nombre_lago,
            "fecha": fecha,
            "umbral_alto_relativo": umbral,
            "promedio": float(np.mean(valores)),
            "mediana": float(np.median(valores)),
            "p90": float(np.percentile(valores, 90)),
            "maximo": float(np.max(valores)),
            "pixeles_validos": int(valores.size),
            "porcentaje_area_valores_altos": porcentaje_alto,
        })

    return pd.DataFrame(registros)


def generar_mapa_persistencia(nombre_lago, carpeta_lago, fechas, umbral, carpeta_salida):
    """Calcula el porcentaje de observaciones en que cada pixel supero el umbral."""
    rutas_validas = [
        ruta_cyano(carpeta_lago, fecha)
        for fecha in fechas
        if os.path.exists(ruta_cyano(carpeta_lago, fecha))
    ]

    if not rutas_validas:
        return

    with rasterio.open(rutas_validas[0]) as ref:
        perfil = ref.profile.copy()
        bounds = ref.bounds

        conteo_altos = np.zeros((ref.height, ref.width), dtype="float32")
        conteo_validos = np.zeros((ref.height, ref.width), dtype="float32")

        for ruta in rutas_validas:
            datos = reproyectar_array_a_referencia(ruta, ref, Resampling.bilinear)
            validos = np.isfinite(datos)
            conteo_validos[validos] += 1
            altos = validos & (datos >= umbral)
            conteo_altos[altos] += 1

        persistencia = np.full(conteo_altos.shape, np.nan, dtype="float32")
        mascara = conteo_validos > 0
        persistencia[mascara] = 100 * conteo_altos[mascara] / conteo_validos[mascara]

    perfil.update(count=1, dtype="float32", nodata=np.nan)

    ruta_tif = os.path.join(
        CARPETA_RASTERS,
        f"persistencia_{nombre_lago.lower()}.tif"
    )

    with rasterio.open(ruta_tif, "w", **perfil) as dst:
        dst.write(persistencia, 1)

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, ax = plt.subplots(figsize=(8, 6))
    imagen = ax.imshow(
        np.ma.masked_invalid(persistencia),
        cmap="magma",
        vmin=0,
        vmax=100,
        extent=extent,
        origin="upper"
    )

    ax.set_title(f"Persistencia de valores altos - {nombre_lago}")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    barra = fig.colorbar(imagen, ax=ax)
    barra.set_label("Observaciones con valor alto (%)")

    fig.tight_layout()

    ruta_png = os.path.join(
        carpeta_salida,
        f"persistencia_{nombre_lago.lower()}.png"
    )

    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[{nombre_lago}] Persistencia guardada -> {ruta_png}")


def generar_boxplot_fechas(nombre_lago, carpeta_lago, fechas, carpeta_salida, max_puntos=6000):
    """Compara la distribucion de cianobacteria entre fechas."""
    rng = np.random.default_rng(42)
    datos_boxplot = []
    etiquetas = []

    for fecha in fechas:
        ruta = ruta_cyano(carpeta_lago, fecha)
        if not os.path.exists(ruta):
            continue

        with rasterio.open(ruta) as src:
            datos = src.read(1).astype("float32")

        valores = datos[np.isfinite(datos)]
        if valores.size == 0:
            continue

        if valores.size > max_puntos:
            indices = rng.choice(valores.size, size=max_puntos, replace=False)
            valores = valores[indices]

        datos_boxplot.append(valores)
        etiquetas.append(fecha)

    if not datos_boxplot:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(etiquetas) * 1.1), 6))
    ax.boxplot(datos_boxplot, tick_labels=etiquetas, showfliers=False)
    ax.set_title(f"Distribucion de cianobacteria por fecha - {nombre_lago}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Clorofila-a estimada")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    ruta = os.path.join(
        carpeta_salida,
        f"boxplot_fechas_{nombre_lago.lower()}.png"
    )

    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generar_mapa_diferencia(nombre_lago, carpeta_lago, df_extension, carpeta_salida):
    """Compara espacialmente la fecha de mayor promedio con la de menor promedio."""
    if df_extension.empty:
        return

    fila_min = df_extension.loc[df_extension["promedio"].idxmin()]
    fila_max = df_extension.loc[df_extension["promedio"].idxmax()]

    fecha_min = pd.to_datetime(fila_min["fecha"]).date().isoformat()
    fecha_max = pd.to_datetime(fila_max["fecha"]).date().isoformat()

    ruta_min = ruta_cyano(carpeta_lago, fecha_min)
    ruta_max = ruta_cyano(carpeta_lago, fecha_max)

    with rasterio.open(ruta_min) as ref:
        datos_min = ref.read(1).astype("float32")
        datos_max = reproyectar_array_a_referencia(ruta_max, ref, Resampling.bilinear)
        bounds = ref.bounds

    mascara = np.isfinite(datos_min) & np.isfinite(datos_max)
    diferencia = np.full(datos_min.shape, np.nan, dtype="float32")
    diferencia[mascara] = datos_max[mascara] - datos_min[mascara]

    valores = diferencia[np.isfinite(diferencia)]
    if valores.size == 0:
        return

    limite = float(np.percentile(np.abs(valores), 98))
    if limite == 0:
        limite = 0.001

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, ax = plt.subplots(figsize=(8, 6))
    imagen = ax.imshow(
        np.ma.masked_invalid(diferencia),
        cmap="coolwarm",
        vmin=-limite,
        vmax=limite,
        extent=extent,
        origin="upper"
    )

    ax.set_title(
        f"Diferencia espacial - {nombre_lago}\n"
        f"{fecha_max} menos {fecha_min}"
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")

    barra = fig.colorbar(imagen, ax=ax)
    barra.set_label("Cambio en clorofila-a estimada")

    fig.tight_layout()

    ruta = os.path.join(
        carpeta_salida,
        f"diferencia_{fecha_max}_menos_{fecha_min}.png"
    )

    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generar_resumen_trimestral(df_extension):
    """Agrupa observaciones por trimestre para explorar un posible patron estacional."""
    df = df_extension.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["trimestre"] = "T" + df["fecha"].dt.quarter.astype(str)

    return (
        df.groupby(["lago", "trimestre"])
        .agg(
            observaciones=("fecha", "count"),
            cyano_promedio=("promedio", "mean"),
            area_alta_promedio_pct=("porcentaje_area_valores_altos", "mean"),
        )
        .reset_index()
    )


def generar_grafica_trimestral(df_trimestral):
    """Guarda una grafica trimestral por lago."""
    for lago, df_lago in df_trimestral.groupby("lago"):
        orden = ["T1", "T2", "T3", "T4"]
        df_lago = (
            df_lago.set_index("trimestre")
            .reindex(orden)
            .dropna(how="all")
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_lago["trimestre"], df_lago["cyano_promedio"], marker="o")
        ax.set_title(f"Patron trimestral exploratorio - {lago}")
        ax.set_xlabel("Trimestre")
        ax.set_ylabel("Promedio de clorofila-a estimada")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        ruta = os.path.join(
            CARPETA_FIGURAS,
            f"patron_trimestral_{lago.lower()}.png"
        )

        fig.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close(fig)


def procesar_lago(nombre_lago, carpeta_lago, fechas, carpeta_salida):
    muestra = obtener_muestra_para_umbral(carpeta_lago, fechas)

    if muestra.size == 0:
        print(f"[{nombre_lago}] No hay datos suficientes.")
        return pd.DataFrame(), np.nan

    umbral = float(np.percentile(muestra, 75))
    print(f"\n[{nombre_lago}] Umbral relativo P75 = {umbral:.2f}")

    df_extension = calcular_extension_por_fecha(
        nombre_lago,
        carpeta_lago,
        fechas,
        umbral
    )

    generar_mapa_persistencia(
        nombre_lago,
        carpeta_lago,
        fechas,
        umbral,
        carpeta_salida
    )

    generar_boxplot_fechas(
        nombre_lago,
        carpeta_lago,
        fechas,
        carpeta_salida
    )

    generar_mapa_diferencia(
        nombre_lago,
        carpeta_lago,
        df_extension,
        carpeta_salida
    )

    return df_extension, umbral


df_atitlan, umbral_atitlan = procesar_lago(
    "Atitlan",
    "data/atitlan",
    fechas_atitlan,
    CARPETA_ATITLAN
)

df_amatitlan, umbral_amatitlan = procesar_lago(
    "Amatitlan",
    "data/amatitlan",
    fechas_amatitlan,
    CARPETA_AMATITLAN
)

df_extension = pd.concat([df_atitlan, df_amatitlan], ignore_index=True)

if not df_extension.empty:
    df_extension["fecha"] = pd.to_datetime(df_extension["fecha"])

    ruta_extension = os.path.join(
        CARPETA_TABLAS,
        "extension_floracion_por_fecha.csv"
    )
    df_extension.to_csv(ruta_extension, index=False)

    df_umbrales = pd.DataFrame([
        {"lago": "Atitlan", "umbral_alto_relativo_p75": umbral_atitlan},
        {"lago": "Amatitlan", "umbral_alto_relativo_p75": umbral_amatitlan},
    ])

    ruta_umbrales = os.path.join(
        CARPETA_TABLAS,
        "umbrales_altos_cianobacteria.csv"
    )
    df_umbrales.to_csv(ruta_umbrales, index=False)

    df_trimestral = generar_resumen_trimestral(df_extension)

    ruta_trimestral = os.path.join(
        CARPETA_TABLAS,
        "resumen_trimestral_cianobacteria.csv"
    )
    df_trimestral.to_csv(ruta_trimestral, index=False)

    generar_grafica_trimestral(df_trimestral)

    print("\nTabla de extension guardada ->", ruta_extension)
    print("Tabla de umbrales guardada ->", ruta_umbrales)
    print("Resumen trimestral guardado ->", ruta_trimestral)
else:
    print("No se generaron resultados exploratorios.")

print("\nAnalisis exploratorio finalizado.")