import os
import math
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# FECHAS OFICIALES
# ============================================================

fechas_atitlan = [
    "2025-01-18",
    "2025-04-13",
    "2025-05-13",
    "2025-07-17",
    "2025-11-21",
    "2025-12-29",
    "2026-02-12",
    "2026-03-24",
    "2026-04-13",
    "2026-04-28",
    "2026-07-22",
]

fechas_amatitlan = [
    "2025-01-28",
    "2025-04-15",
    "2025-04-28",
    "2025-11-24",
    "2026-01-08",
    "2026-02-02",
    "2026-02-07",
    "2026-03-29",
    "2026-04-13",
    "2026-04-28",
    "2026-06-19",
]


# ============================================================
# CARPETAS DE RESULTADOS
# ============================================================

CARPETA_TABLAS = "resultados/tablas"
CARPETA_FIGURAS = "resultados/figuras/espacial"

CARPETA_FIGURAS_ATITLAN = os.path.join(
    CARPETA_FIGURAS,
    "atitlan"
)

CARPETA_FIGURAS_AMATITLAN = os.path.join(
    CARPETA_FIGURAS,
    "amatitlan"
)


def crear_carpetas():
    """
    Crea automaticamente todas las carpetas necesarias.
    """

    carpetas = [
        CARPETA_TABLAS,
        CARPETA_FIGURAS,
        CARPETA_FIGURAS_ATITLAN,
        CARPETA_FIGURAS_AMATITLAN,
    ]

    for carpeta in carpetas:
        os.makedirs(carpeta, exist_ok=True)


# ============================================================
# CARGA DEL RASTER
# ============================================================

def cargar_cianobacteria(carpeta_lago, fecha):
    """
    Carga el raster de cianobacteria de una fecha.

    Devuelve:
        datos
        extent
    """

    ruta = os.path.join(
        carpeta_lago,
        f"{fecha}_cyano.tif"
    )

    if not os.path.exists(ruta):
        print(f"No existe {ruta}. Se omite.")
        return None, None

    with rasterio.open(ruta) as src:

        datos = src.read(1).astype("float32")

        bounds = src.bounds

        extent = [
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
        ]

    # Cualquier valor infinito se convierte en NaN
    datos[~np.isfinite(datos)] = np.nan

    return datos, extent


# ============================================================
# ESCALA COMUN PARA LOS MAPAS
# ============================================================

def calcular_escala_comun(carpeta_lago, fechas):
    """
    Calcula una escala comun para todos los mapas de un lago.

    Esto permite comparar visualmente diferentes fechas
    usando la misma escala de colores.
    """

    todos_los_valores = []

    for fecha in fechas:

        datos, _ = cargar_cianobacteria(
            carpeta_lago,
            fecha
        )

        if datos is None:
            continue

        valores_validos = datos[np.isfinite(datos)]

        if valores_validos.size > 0:
            todos_los_valores.append(valores_validos)

    if not todos_los_valores:
        return None, None

    todos_los_valores = np.concatenate(
        todos_los_valores
    )

    # Se usan percentiles para reducir el efecto
    # de valores extremos.
    vmin = np.percentile(
        todos_los_valores,
        2
    )

    vmax = np.percentile(
        todos_los_valores,
        98
    )

    # Evitar problemas si todos los valores son iguales
    if vmin == vmax:
        vmax = vmin + 0.001

    return vmin, vmax


# ============================================================
# 5.1 MAPAS INDIVIDUALES
# ============================================================

def generar_mapas_individuales(
    nombre_lago,
    carpeta_lago,
    fechas,
    carpeta_salida,
    vmin,
    vmax
):
    """
    Genera un mapa de cianobacteria por cada fecha.
    """

    for fecha in fechas:

        datos, extent = cargar_cianobacteria(
            carpeta_lago,
            fecha
        )

        if datos is None:
            continue

        if np.all(np.isnan(datos)):
            print(
                f"[{nombre_lago}] {fecha} "
                "no tiene pixeles validos."
            )
            continue

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        imagen = ax.imshow(
            np.ma.masked_invalid(datos),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="upper"
        )

        ax.set_title(
            f"Distribucion espacial de cianobacteria\n"
            f"{nombre_lago} - {fecha}"
        )

        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")

        barra = fig.colorbar(
            imagen,
            ax=ax
        )

        barra.set_label(
            "Clorofila-a estimada"
        )

        fig.tight_layout()

        ruta_salida = os.path.join(
            carpeta_salida,
            f"{fecha}.png"
        )

        fig.savefig(
            ruta_salida,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"[{nombre_lago}] "
            f"Mapa guardado -> {ruta_salida}"
        )


# ============================================================
# 5.2 MAPA COMPARATIVO
# ============================================================

def generar_mapa_comparativo(
    nombre_lago,
    carpeta_lago,
    fechas,
    carpeta_salida,
    vmin,
    vmax
):
    """
    Coloca todas las fechas disponibles en una misma figura
    para facilitar la comparacion espacial y temporal.
    """

    imagenes = []

    for fecha in fechas:

        datos, extent = cargar_cianobacteria(
            carpeta_lago,
            fecha
        )

        if datos is None:
            continue

        if np.all(np.isnan(datos)):
            continue

        imagenes.append(
            (fecha, datos, extent)
        )

    if not imagenes:
        print(
            f"[{nombre_lago}] "
            "No existen datos para generar comparacion."
        )
        return

    columnas = 3

    filas = math.ceil(
        len(imagenes) / columnas
    )

    fig, axes = plt.subplots(
        filas,
        columnas,
        figsize=(15, 4.5 * filas),
        constrained_layout=True
    )

    # Convertir axes siempre a arreglo plano
    axes = np.array(axes).reshape(-1)

    ultima_imagen = None

    for i, (fecha, datos, extent) in enumerate(imagenes):

        ax = axes[i]

        ultima_imagen = ax.imshow(
            np.ma.masked_invalid(datos),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="upper"
        )

        ax.set_title(fecha)

        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")

    # Ocultar cuadros sobrantes
    for j in range(
        len(imagenes),
        len(axes)
    ):
        axes[j].axis("off")

    fig.suptitle(
        f"Comparacion espacial de cianobacteria - {nombre_lago}",
        fontsize=16
    )

    if ultima_imagen is not None:

        barra = fig.colorbar(
            ultima_imagen,
            ax=axes.tolist(),
            shrink=0.75
        )

        barra.set_label(
            "Clorofila-a estimada"
        )

    nombre_archivo = (
        f"comparacion_espacial_"
        f"{nombre_lago.lower()}.png"
    )

    ruta_salida = os.path.join(
        carpeta_salida,
        nombre_archivo
    )

    fig.savefig(
        ruta_salida,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[{nombre_lago}] "
        f"Comparacion guardada -> {ruta_salida}"
    )


# ============================================================
# ESTADISTICAS ESPACIALES
# ============================================================

def calcular_resumen_espacial(
    nombre_lago,
    carpeta_lago,
    fechas
):
    """
    Calcula estadisticas de los valores encontrados
    dentro del lago para cada fecha.
    """

    registros = []

    for fecha in fechas:

        datos, _ = cargar_cianobacteria(
            carpeta_lago,
            fecha
        )

        if datos is None:
            continue

        valores = datos[
            np.isfinite(datos)
        ]

        if valores.size == 0:

            registros.append({
                "lago": nombre_lago,
                "fecha": fecha,
                "promedio": np.nan,
                "mediana": np.nan,
                "p90": np.nan,
                "maximo": np.nan,
                "pixeles_validos": 0,
                "porcentaje_pixeles_validos": 0
            })

            continue

        porcentaje_validos = (
            valores.size /
            datos.size
        ) * 100

        registros.append({
            "lago": nombre_lago,
            "fecha": fecha,
            "promedio": float(
                np.mean(valores)
            ),
            "mediana": float(
                np.median(valores)
            ),
            "p90": float(
                np.percentile(valores, 90)
            ),
            "maximo": float(
                np.max(valores)
            ),
            "pixeles_validos": int(
                valores.size
            ),
            "porcentaje_pixeles_validos":
                float(porcentaje_validos)
        })

    return pd.DataFrame(registros)


# ============================================================
# PROCESAMIENTO COMPLETO DE CADA LAGO
# ============================================================

def procesar_lago(
    nombre_lago,
    carpeta_lago,
    fechas,
    carpeta_salida
):
    """
    Ejecuta todo el analisis espacial para un lago.
    """

    print("\n")
    print("=" * 60)
    print(
        f"ANALISIS ESPACIAL - {nombre_lago.upper()}"
    )
    print("=" * 60)

    # Calcular escala comun
    vmin, vmax = calcular_escala_comun(
        carpeta_lago,
        fechas
    )

    if vmin is None or vmax is None:

        print(
            f"No hay datos suficientes "
            f"para {nombre_lago}."
        )

        return pd.DataFrame()

    print(
        f"Escala visual comun: "
        f"{vmin:.2f} - {vmax:.2f}"
    )

    # 5.1
    generar_mapas_individuales(
        nombre_lago,
        carpeta_lago,
        fechas,
        carpeta_salida,
        vmin,
        vmax
    )

    # 5.2
    generar_mapa_comparativo(
        nombre_lago,
        carpeta_lago,
        fechas,
        carpeta_salida,
        vmin,
        vmax
    )

    # Estadisticas
    resumen = calcular_resumen_espacial(
        nombre_lago,
        carpeta_lago,
        fechas
    )

    return resumen


# ============================================================
# EJECUCION
# ============================================================

crear_carpetas()


resumen_atitlan = procesar_lago(
    "Atitlan",
    "data/atitlan",
    fechas_atitlan,
    CARPETA_FIGURAS_ATITLAN
)


resumen_amatitlan = procesar_lago(
    "Amatitlan",
    "data/amatitlan",
    fechas_amatitlan,
    CARPETA_FIGURAS_AMATITLAN
)


# Unir resultados de ambos lagos
df_resumen_espacial = pd.concat(
    [
        resumen_atitlan,
        resumen_amatitlan
    ],
    ignore_index=True
)


# Convertir fecha a formato de fecha real
if not df_resumen_espacial.empty:

    df_resumen_espacial["fecha"] = pd.to_datetime(
        df_resumen_espacial["fecha"]
    )

    ruta_tabla = os.path.join(
        CARPETA_TABLAS,
        "resumen_espacial_cianobacteria.csv"
    )

    df_resumen_espacial.to_csv(
        ruta_tabla,
        index=False
    )

    print("\n")
    print("=" * 60)
    print("RESUMEN ESPACIAL")
    print("=" * 60)

    print(
        df_resumen_espacial.to_string(
            index=False
        )
    )

    print(
        f"\nTabla guardada -> {ruta_tabla}"
    )


print("\nAnalisis espacial finalizado.")