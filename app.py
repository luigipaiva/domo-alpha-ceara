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

# --- 1. SETUP ---
st.set_page_config(layout="wide", page_title="DOMO Alpha - Dashboard")

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

# --- 2. AUXILIARES ---
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
    """Calcula a área em hectares de uma máscara binária (0 ou 1)"""
    pixel_area = ee.Image.pixelArea().updateMask(image_mask)
    stats = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True
    )
    area_m2 = stats.getInfo().get('area', 0)
    return area_m2 / 10000  # Converte m² para Hectares

# Reset Session
for key in ['roi', 'domo_map', 'roi_name', 'map_bounds', 'legend_title', 'sat_source', 'map_key', 'metric_area']:
    if key not in st.session_state: st.session_state[key] = None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("📊 DOMO Analytics")
    
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
            with st.spinner("Gerando geometria..."):
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

# --- 4. DASHBOARD ---
if st.session_state['roi'] and connected:
    
    # 4.1 - Painel de Métricas (Topo)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Municípios Alvo", len(selecao) if selecao else 0)
    with col2:
        val = st.session_state['metric_area']
        label_metric = "Área Detectada"
        if "Água" in modo: label_metric = "Espelho D'água Atual"
        elif "Queimada" in modo: label_metric = "Área Queimada"
        
        st.metric(label_metric, f"{val:.2f} ha" if val is not None else "--")
    with col3:
        st.metric("Satélite Ativo", st.session_state['sat_source'] if st.session_state['sat_source'] else "Aguardando")

    # 4.2 - Mapa
    m = geemap.Map()
    if st.session_state['map_bounds']: m.fit_bounds(st.session_state['map_bounds'])
    
    if st.button(f"⚡ CALCULAR E MAPEAR: {modo.split('(')[0]}"):
        roi = st.session_state['roi']
        vis_params = {}
        layer_name = ""
        mask_final = None # Guarda a máscara para cálculo de área
        scale_calc = 30 # Default Landsat
        
        # --- LÓGICA LANDSAT (Água) ---
        if "Landsat" in modo:
            with st.spinner("Processando Landsat 9..."):
                l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi)
                img = l9.filter(ee.Filter.lt('CLOUD_COVER', 15)).sort('system:time_start', False).first()
                if img:
                    green = img.select('SR_B3').multiply(0.0000275).add(-0.2)
                    swir = img.select('SR_B6').multiply(0.0000275).add(-0.2)
                    mndwi = green.subtract(swir).divide(green.add(swir))
                    
                    mask_final = mndwi.gt(0.0) # Máscara binária
                    st.session_state['domo_map'] = mndwi.updateMask(mask_final).clip(roi)
                    
                    st.session_state['sat_source'] = "Landsat 9 (30m)"
                    vis_params = {'min': 0, 'max': 0.6, 'palette': ['white', 'blue', 'navy']}
                    layer_name = "Água (Landsat)"
                    scale_calc = 30
                else: st.warning("Sem imagem Landsat.")

        # --- LÓGICA SENTINEL (Outros) ---
        else:
            with st.spinner("Processando Sentinel-2..."):
                s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
                # Seleciona bandas necessárias
                s2 = s2.select(['B3', 'B4', 'B5', 'B8', 'B11', 'B12'])
                img = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).sort('system:time_start', False).first()
                
                if img:
                    st.session_state['sat_source'] = "Sentinel-2 (10m)"
                    scale_calc = 10
                    
                    if "Clorofila" in modo:
                        # NDCI
                        ndci = img.normalizedDifference(['B5', 'B4'])
                        # Water Mask (Sentinel MNDWI)
                        water_mask = img.normalizedDifference(['B3', 'B11']).gt(-0.1)
                        
                        mask_final = water_mask # Calculamos a área de água com potencial de algas
                        st.session_state['domo_map'] = ndci.updateMask(water_mask).clip(roi)
                        vis_params = {'min': 0.0, 'max': 0.15, 'palette': ['blue', 'cyan', 'lime', 'yellow', 'red']}
                        layer_name = "Risco Algas (NDCI)"

                    elif "Desmatamento" in modo:
                        ref_hist = s2.filterDate('2023-01-01', '2024-12-31').median()
                        ndvi_now = img.normalizedDifference(['B8','B4'])
                        ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
                        # Lógica de Supressão
                        alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
                        mask_final = alerta.connectedPixelCount(30).gte(15)
                        
                        st.session_state['domo_map'] = alerta.updateMask(mask_final).clip(roi)
                        vis_params = {'palette': ['red']}
                        layer_name = "Supressão Veg."

                    elif "Queimadas" in modo:
                        # NBR = (NIR - SWIR) / (NIR + SWIR)
                        nbr = img.normalizedDifference(['B8', 'B12'])
                        mask_final = nbr.lt(-0.1) # Cicatriz de fogo
                        
                        st.session_state['domo_map'] = nbr.updateMask(mask_final).clip(roi)
                        vis_params = {'min': -0.5, 'max': -0.1, 'palette': ['black', 'orange', 'red']}
                        layer_name = "Cicatrizes Fogo"

                else: st.warning("Sem imagem Sentinel.")

        # CÁLCULO DE ÁREA (O Cérebro do Dashboard)
        if mask_final is not None:
            area_ha = calculate_hectares(mask_final, roi, scale_calc)
            st.session_state['metric_area'] = area_ha
            st.session_state['legend_title'] = layer_name
            st.session_state['vis_params'] = vis_params
            st.rerun() # Recarrega a página para atualizar os números lá em cima

    # Renderiza Mapa
    if st.session_state['domo_map']:
        vis = st.session_state.get('vis_params', {})
        name = st.session_state.get('legend_title', 'Result')
        m.addLayer(st.session_state['domo_map'], vis, name)
        
        if "Landsat" in modo: m.add_colorbar(vis, label="Profundidade")
        elif "Clorofila" in modo: m.add_colorbar(vis, label="Conc. Algas")

    key_mapa = st.session_state.get('map_key', 'map_default')
    m.to_streamlit(height=550, key=key_mapa)
