import os
import numpy as np
import rasterio
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
from rasterio.transform import from_bounds



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


# ============================================================
# COLECCION SENTINEL-2 PARA COPERNICUS DATA SPACE
# ============================================================

# Las colecciones predefinidas de sentinelhub pueden apuntar
# al servidor tradicional de Sentinel Hub.
# Creamos una variante que utilice explicitamente CDSE.
SENTINEL2_L1C_CDSE = DataCollection.SENTINEL2_L1C.define_from(
    "SENTINEL2_L1C_CDSE",
    service_url=config.sh_base_url
)

print("Configuracion Sentinel Hub:")
print(f"  Base URL: {config.sh_base_url}")
print(f"  Token URL: {config.sh_token_url}")
print(f"  Client ID cargado: {bool(config.sh_client_id)}")
print(f"  Client Secret cargado: {bool(config.sh_client_secret)}")



lago_atitlan = {
    "west": -91.326256,
    "east": -91.07151,
    "south": 14.5948,
    "north": 14.750979
}

lago_amatitlan = {
    "west": -90.638065,
    "east": -90.512924,
    "south": 14.412347,
    "north": 14.493799
}

# fechas_atitlan = [
#     "2025-01-18", "2025-04-13", "2025-05-13", 
# ]
# """ "2025-07-17",
#     "2025-11-21", "2025-12-29", "2026-02-12", "2026-03-24",
#     "2026-04-13", "2026-04-28", "2026-07-22", """

# fechas_amatitlan = [
#     "2025-01-28", "2025-04-15", "2025-04-28", 
# ]

# """ "2025-11-24",
#     "2026-01-08", "2026-02-02", "2026-02-07", "2026-03-29",
#     "2026-04-13", "2026-04-28", "2026-06-19", """

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

def calcular_ndvi_ndwi(ruta_tif):
    with rasterio.open(ruta_tif) as src:
        b03 = src.read(1).astype("float32")  # verde
        b04 = src.read(2).astype("float32")  # rojo
        b08 = src.read(3).astype("float32")  # NIR
        perfil = src.profile

    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (b08 - b04) / (b08 + b04)
    ndwi = (b03 - b08) / (b03 + b08)

    return ndvi, ndwi, perfil

def guardar_raster(array, perfil, ruta_salida):
    perfil_out = perfil.copy()
    perfil_out.update(count=1, dtype="float32")
    with rasterio.open(ruta_salida, "w", **perfil_out) as dst:
        dst.write(array.astype("float32"), 1)

def procesar_indices_locales(carpeta_lago, fechas):
    for fecha in fechas:
        ruta_tif = os.path.join(carpeta_lago, f"{fecha}.tif")
        if not os.path.exists(ruta_tif):
            print(f"Falta el archivo {ruta_tif}, se omite.")
            continue

        ndvi, ndwi, perfil = calcular_ndvi_ndwi(ruta_tif)

        guardar_raster(ndvi, perfil, os.path.join(carpeta_lago, f"{fecha}_ndvi.tif"))
        guardar_raster(ndwi, perfil, os.path.join(carpeta_lago, f"{fecha}_ndwi.tif"))
        print(f"NDVI/NDWI calculados para {fecha}")

EVALSCRIPT_CIANOBACTERIA = """
//VERSION=3
function setup() {
return {
    input: ["B02","B03","B04","B05","B07","B08","B8A","B11","B12"],
    output: { bands: 1, sampleType: "FLOAT32" }
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

function NDCI(a,b) { return (b-a)/(b+a); }
let NDCIv = NDCI(s.B04, s.B05);
let chl = 826.57*Math.pow(NDCIv,3) - 176.43*Math.pow(NDCIv,2) + 19*NDCIv + 4.071;

// Fuera de agua -> NaN, para no contaminar el promedio del indice
if (water == 0) { return [NaN]; }
return [chl];
}
"""

def descargar_cianobacteria(bbox_dict, fecha, ruta_salida, resolucion=20):
    bbox = BBox(
        bbox=[bbox_dict["west"], bbox_dict["south"], bbox_dict["east"], bbox_dict["north"]],
        crs=CRS.WGS84,
    )
    ancho, alto = bbox_to_dimensions(bbox, resolution=resolucion)
 
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_CIANOBACTERIA,
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

    # Si Sentinel Hub devuelve H x W x 1, nos quedamos con una sola banda
    if resultado.ndim == 3:
        resultado = resultado[:, :, 0]

    alto, ancho = resultado.shape

    # Relacionamos los píxeles con las coordenadas reales del lago
    transform = from_bounds(
        bbox_dict["west"],
        bbox_dict["south"],
        bbox_dict["east"],
        bbox_dict["north"],
        ancho,
        alto,
    )

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    with rasterio.open(
        ruta_salida,
        "w",
        driver="GTiff",
        height=alto,
        width=ancho,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(resultado.astype("float32"), 1)

    print(f"Indice de cianobacteria guardado -> {ruta_salida}")

def procesar_cianobacteria(nombre_lago, bbox_dict, fechas, carpeta_lago):
    for fecha in fechas:
        ruta_salida = os.path.join(carpeta_lago, f"{fecha}_cyano.tif")
        if os.path.exists(ruta_salida):
            print(f"[{nombre_lago}] {fecha} cyano ya existe, se omite.")
            continue
        try:
            descargar_cianobacteria(bbox_dict, fecha, ruta_salida)
        except Exception as e:
            print(f"[{nombre_lago}] ERROR cyano en {fecha}: {e}")

procesar_indices_locales("data/atitlan", fechas_atitlan)
procesar_indices_locales("data/amatitlan", fechas_amatitlan)

procesar_cianobacteria("Atitlan", lago_atitlan, fechas_atitlan, "data/atitlan")
procesar_cianobacteria("Amatitlan", lago_amatitlan, fechas_amatitlan, "data/amatitlan")
