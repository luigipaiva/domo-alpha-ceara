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
import uuid

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(layout="wide", page_title="DOMO Alpha - Turbo")

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
        st.error(f"❌ Erro EE: {e}")
        return False

connected = init_ee_enterprise()

@st.cache_resource
def load_model():
    try: return genai.GenerativeModel('gemini-1.5-flash')
    except: return None
model = load_model()

# --- 2. CACHE INTELIGENTE ---
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
    if not geoms: return None
    return unary_union(geoms).buffer(0)

def calculate_hectares(image_mask, geometry, scale=30):
    """Cálculo otimizado de área"""
    pixel_area = ee.Image.pixelArea().updateMask(image_mask)
    # bestEffort=True permite que o Google ajuste a escala se a área for muito grande
    stats = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e10, # Aumentado limite de pixels
        bestEffort=True
    )
    area_m2 = stats.getInfo().get('area', 0)
    return area_m2 / 10000

# Reset Session
for key in ['roi', 'domo_map', 'roi_name', 'map_bounds', 'legend_title', 'sat_source', 'map_key', 'metric_area']:
    if key not in st.session_state: st.session_state[key] = None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚡ DOMO Turbo")
    
    modo = st.radio(
        "Monitoramento:",
        ["💧 Espelho D'água (Landsat)", "🧪 Clorofila (Sentinel)", "🌳 Desmatamento (Sentinel)", "🔥 Queimadas (Sentinel)"]
    )
    
    st.divider()
    try:
        municipios = get_ceara_cities()
        selecao = st.multiselect("Municípios", [m['nome'] for m in municipios])
    except: st.error("Erro IBGE")

    if st.button("📍 CARREGAR ÁREA", type="primary", disabled=not connected):
        if selecao:
            with st.spinner("Geometria..."):
                st.session_state['domo_map'] = None
                st.session_state['metric_area'] = None
                
                ids = [m['id'] for m in municipios if m['nome'] in selecao]
                geom = get_fast_geometry(ids)
                
                if geom:
                    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
                    st.session_state['roi'] = geemap.geopandas_to_ee(gdf).geometry()
                    st.session_state['roi_name'] = ", ".join(selecao)
                    b = st.session_state['roi'].bounds().getInfo()['coordinates'][0]
                    st.session_state['map_bounds'] = [[min([p[1] for p in b]), min([p[0] for p in b])], [max([p[1] for p in b]), max([p[0] for p in b])]]
                    st.session_state['map_key'] = str(uuid.uuid4())
                    st.success(f"Alvo: {st.session_state['roi_name']}")

# --- 4. MOTOR ---
if st.session_state['roi'] and connected:
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Municípios", len(selecao) if selecao else 0)
    col2.metric("Área Detectada", f"{st.session_state['metric_area']:.2f} ha" if st.session_state['metric_area'] is not None else "--")
    col3.metric("Satélite", st.session_state['sat_source'] if st.session_state['sat_source'] else "--")

    m = geemap.Map()
    if st.session_state['map_bounds']: m.fit_bounds(st.session_state['map_bounds'])
    
    if st.button(f"🚀 EXECUTAR: {modo.split('(')[0]}"):
        roi = st.session_state['roi']
        vis_params = {}
        layer_name = ""
        mask_final = None
        scale_calc = 30
        
        # --- MOTOR LANDSAT ---
        if "Landsat" in modo:
            with st.spinner("Landsat 9..."):
                l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi)
                img = l9.filter(ee.Filter.lt('CLOUD_COVER', 15)).sort('system:time_start', False).first()
                if img:
                    green = img.select('SR_B3').multiply(0.0000275).add(-0.2)
                    swir = img.select('SR_B6').multiply(0.0000275).add(-0.2)
                    mndwi = green.subtract(swir).divide(green.add(swir))
                    mask_final = mndwi.gt(0.0)
                    st.session_state['domo_map'] = mndwi.updateMask(mask_final).clip(roi)
                    st.session_state['sat_source'] = "Landsat 9"
                    vis_params = {'min': 0, 'max': 0.6, 'palette': ['white', 'blue', 'navy']}
                    layer_name = "Água"
                else: st.warning("Sem Landsat limpo.")

        # --- MOTOR SENTINEL (Otimizado) ---
        else:
            with st.spinner("Sentinel-2..."):
                s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
                img = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).sort('system:time_start', False).first()
                
                if img:
                    st.session_state['sat_source'] = "Sentinel-2"
                    
                    if "Desmatamento" in modo:
                        # [OTIMIZAÇÃO CRÍTICA]
                        # Em vez de 2 anos de histórico, pega apenas 3 meses do ano anterior
                        # Ex: Se hoje é Jan/2025, comparamos com Nov/23 a Jan/24
                        date_now = img.date()
                        start_hist = date_now.advance(-13, 'month')
                        end_hist = date_now.advance(-11, 'month')
                        
                        # Processa apenas ~5 imagens históricas (MUITO mais rápido)
                        ref_hist = s2.filterDate(start_hist, end_hist).median()
                        
                        ndvi_now = img.normalizedDifference(['B8','B4'])
                        ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
                        
                        # Supressão: Era >0.45 e virou <0.2
                        alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
                        mask_final = alerta.connectedPixelCount(30).gte(15)
                        
                        st.session_state['domo_map'] = alerta.updateMask(mask_final).clip(roi)
                        vis_params = {'palette': ['red']}
                        layer_name = "Supressão"
                        scale_calc = 20 # Cálculo 4x mais rápido que 10m

                    elif "Clorofila" in modo:
                        scale_calc = 20
                        water_mask = img.normalizedDifference(['B3', 'B11']).gt(-0.1)
                        ndci = img.normalizedDifference(['B5', 'B4'])
                        mask_final = water_mask
                        st.session_state['domo_map'] = ndci.updateMask(water_mask).clip(roi)
                        vis_params = {'min': 0, 'max': 0.15, 'palette': ['blue', 'lime', 'red']}
                        layer_name = "Algas"

                    elif "Queimadas" in modo:
                        scale_calc = 20
                        nbr = img.normalizedDifference(['B8', 'B12'])
                        mask_final = nbr.lt(-0.1)
                        st.session_state['domo_map'] = nbr.updateMask(mask_final).clip(roi)
                        vis_params = {'min': -0.5, 'max': -0.1, 'palette': ['black', 'orange', 'red']}
                        layer_name = "Fogo"
                else: st.warning("Sem Sentinel limpo.")

        if mask_final is not None:
            # Cálculo de área com bestEffort=True (evita timeout em áreas grandes)
            area_ha = calculate_hectares(mask_final, roi, scale_calc)
            st.session_state['metric_area'] = area_ha
            st.session_state['legend_title'] = layer_name
            st.session_state['vis_params'] = vis_params
            st.rerun()

    if st.session_state['domo_map']:
        vis = st.session_state.get('vis_params', {})
        name = st.session_state.get('legend_title', 'Result')
        m.addLayer(st.session_state['domo_map'], vis, name)
        
        if "Desmatamento" in modo: m.add_legend(title="Legenda", labels=["Desmatamento"], colors=["#FF0000"])
        elif "Landsat" in modo: m.add_colorbar(vis, label="Profundidade")
        elif "Clorofila" in modo: m.add_colorbar(vis, label="Algas")

    key_mapa = st.session_state.get('map_key', 'map_default')
    m.to_streamlit(height=600, key=key_mapa)


