import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import geopandas as gpd
import requests
import google.generativeai as genai
from shapely.geometry import shape
from shapely.ops import unary_union
import json

# --- 1. SETUP E CONEXÃO ---
st.set_page_config(layout="wide", page_title="DOMO Híbrido - Landsat/Sentinel")

PROJECT_ID = st.secrets.get("PROJECT_ID", "domo-alpha-ia")
API_KEY = st.secrets.get("API_KEY")
EE_KEYS_RAW = st.secrets.get("EE_KEYS")

if API_KEY: genai.configure(api_key=API_KEY)

@st.cache_resource
def init_ee_enterprise():
    try:
        if EE_KEYS_RAW:
            key_dict = json.loads(EE_KEYS_RAW, strict=False)
            if 'private_key' in key_dict:
                key_dict['private_key'] = key_dict['private_key'].replace('\\n', '\n')
            credentials = ee.ServiceAccountCredentials(
                key_dict['client_email'], key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials, project=PROJECT_ID)
            return True
        else:
            ee.Initialize(project=PROJECT_ID)
            return True
    except Exception as e:
        st.error(f"❌ Erro: {e}")
        return False

connected = init_ee_enterprise()

@st.cache_resource
def load_model():
    try: return genai.GenerativeModel('gemini-1.5-flash')
    except: return None
model = load_model()

# --- 2. CACHE GEOGRÁFICO ---
@st.cache_data(ttl=86400)
def get_ceara_cities():
    return requests.get("https://servicodados.ibge.gov.br/api/v1/localidades/estados/23/municipios?orderBy=nome").json()

@st.cache_data(ttl=86400)
def get_fast_geometry(mun_ids):
    geoms = []
    for m_id in mun_ids:
        url = f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{m_id}?formato=application/vnd.geo+json&qualidade=minima"
        try:
            data = requests.get(url).json()
            g = shape(data['features'][0]['geometry']) if 'features' in data else shape(data['geometry'])
            geoms.append(g.simplify(0.005))
        except: continue
    return unary_union(geoms).buffer(0)

for key in ['roi', 'domo_map', 'roi_name', 'map_bounds', 'legend_title', 'sat_source']:
    if key not in st.session_state: st.session_state[key] = None

# --- 3. INTERFACE ---
with st.sidebar:
    st.title("🛰️ DOMO Multi-Sensor")
    st.caption("Fusão Landsat 9 + Sentinel-2")
    
    modo = st.radio(
        "Objetivo da Análise:",
        ["💧 Espelho D'água (Landsat 9)", "🧪 Clorofila/Algas (Sentinel-2)", "🌳 Desmatamento (Sentinel-2)"]
    )
    
    st.divider()
    try:
        municipios = get_ceara_cities()
        selecao = st.multiselect("Municípios", [m['nome'] for m in municipios])
    except: st.error("Erro IBGE")

    if st.button("📍 CARREGAR ÁREA", type="primary", disabled=not connected):
        if selecao:
            with st.spinner("Definindo geometria..."):
                ids = [m['id'] for m in municipios if m['nome'] in selecao]
                geom = get_fast_geometry(ids)
                gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
                st.session_state['roi'] = geemap.geopandas_to_ee(gdf).geometry()
                st.session_state['roi_name'] = ", ".join(selecao)
                b = st.session_state['roi'].bounds().getInfo()['coordinates'][0]
                st.session_state['map_bounds'] = [[min([p[1] for p in b]), min([p[0] for p in b])], [max([p[1] for p in b]), max([p[0] for p in b])]]
                st.success("Pronto.")

# --- 4. MOTOR HÍBRIDO ---
if st.session_state['roi'] and connected:
    m = geemap.Map()
    if st.session_state['map_bounds']: m.fit_bounds(st.session_state['map_bounds'])
    
    if st.button(f"⚡ EXECUTAR: {modo.split('(')[0]}"):
        roi = st.session_state['roi']
        vis_params = {}
        layer_name = ""
        
        # --- MOTOR 1: LANDSAT 9 (PARA ÁGUA) ---
        if "Landsat" in modo:
            with st.spinner("Acessando Landsat 9 OLI-2..."):
                # Coleção Landsat 9 Level 2 (Reflectância de Superfície)
                l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi)
                img = l9.filter(ee.Filter.lt('CLOUD_COVER', 15)).sort('system:time_start', False).first()
                
                if img:
                    # Landsat Bands: SR_B3 (Green), SR_B6 (SWIR 1)
                    # MNDWI = (Green - SWIR) / (Green + SWIR)
                    green = img.select('SR_B3').multiply(0.0000275).add(-0.2)
                    swir = img.select('SR_B6').multiply(0.0000275).add(-0.2)
                    
                    mndwi = green.subtract(swir).divide(green.add(swir))
                    
                    # Máscara de água (Landsat costuma ser bem preciso com > 0)
                    st.session_state['domo_map'] = mndwi.updateMask(mndwi.gt(0.0)).clip(roi)
                    st.session_state['sat_source'] = "Landsat 9 (30m)"
                    vis_params = {'min': 0, 'max': 0.6, 'palette': ['white', 'blue', 'navy']}
                    layer_name = "Espelho D'água (Landsat)"
                else: st.warning("Sem imagem Landsat limpa recente.")

        # --- MOTOR 2: SENTINEL-2 (PARA CLOROFILA E MATAS) ---
        else:
            with st.spinner("Acessando Sentinel-2 MSI..."):
                s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
                img = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).sort('system:time_start', False).first()
                
                if img:
                    st.session_state['sat_source'] = "Sentinel-2 (10m)"
                    
                    if "Clorofila" in modo:
                        # NDCI = (RedEdge - Red) / (RedEdge + Red)
                        # Sentinel Bands: B5 (Red Edge 1), B4 (Red)
                        ndci = img.normalizedDifference(['B5', 'B4'])
                        
                        # Usamos MNDWI do Sentinel apenas para recortar onde é água
                        mask_water = img.normalizedDifference(['B3', 'B11']).gt(-0.1)
                        
                        st.session_state['domo_map'] = ndci.updateMask(mask_water).clip(roi)
                        vis_params = {'min': 0.0, 'max': 0.15, 'palette': ['blue', 'cyan', 'lime', 'yellow', 'red']}
                        layer_name = "NDCI (Algas)"
                        
                    elif "Desmatamento" in modo:
                        ref_hist = s2.filterDate('2023-01-01', '2024-12-31').median()
                        ndvi_now = img.normalizedDifference(['B8','B4'])
                        ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
                        alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
                        st.session_state['domo_map'] = alerta.updateMask(alerta.connectedPixelCount(30).gte(15)).clip(roi)
                        vis_params = {'palette': ['red']}
                        layer_name = "Desmatamento"
                else: st.warning("Sem imagem Sentinel limpa recente.")

        st.session_state['legend_title'] = layer_name
        st.session_state['vis_params'] = vis_params

    # Exibição
    if st.session_state['domo_map']:
        st.info(f"Fonte do Dado: {st.session_state.get('sat_source')}")
        vis = st.session_state.get('vis_params', {})
        name = st.session_state.get('legend_title', 'Layer')
        m.addLayer(st.session_state['domo_map'], vis, name)
        
        if "Landsat" in modo:
             m.add_colorbar(vis, label="Profundidade/Água (Landsat)")
        elif "Clorofila" in modo:
             m.add_colorbar(vis, label="Concentração de Algas (Sentinel)")

    m.to_streamlit(height=600)
