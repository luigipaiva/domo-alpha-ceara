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

# --- 1. SETUP E AUTENTICAÇÃO ---
st.set_page_config(layout="wide", page_title="DOMO Alpha Earth - Hídrico")

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
        st.error(f"❌ Erro de Autenticação: {e}")
        return False

connected = init_ee_enterprise()

@st.cache_resource
def load_model():
    try: return genai.GenerativeModel('gemini-1.5-flash')
    except: return None
model = load_model()

# --- 2. DADOS E GEOMETRIA ---
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

for key in ['roi', 'domo_map', 'roi_name', 'map_bounds', 'last_scan_data', 'legend_title', 'img_date']:
    if key not in st.session_state: st.session_state[key] = None

# --- 3. INTERFACE ---
with st.sidebar:
    st.title("🌵 DOMO Alpha Earth")
    if connected: st.success("🛰️ Sistema Online")
    
    st.divider()
    modo = st.radio(
        "Lente de Análise:",
        ["🌳 Desmatamento", "💧 Espelho D'água (MNDWI)", "🧪 Qualidade (NDCI)", "🔥 Queimadas (NBR)"]
    )
    
    st.divider()
    try:
        municipios = get_ceara_cities()
        selecao = st.multiselect("Municípios", [m['nome'] for m in municipios])
    except: st.error("Erro IBGE")

    if st.button("📍 CARREGAR ÁREA", type="primary", use_container_width=True, disabled=not connected):
        if selecao:
            with st.spinner("Carregando polígono..."):
                ids = [m['id'] for m in municipios if m['nome'] in selecao]
                geom = get_fast_geometry(ids)
                gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
                st.session_state['roi'] = geemap.geopandas_to_ee(gdf).geometry()
                st.session_state['roi_name'] = ", ".join(selecao)
                b = st.session_state['roi'].bounds().getInfo()['coordinates'][0]
                st.session_state['map_bounds'] = [[min([p[1] for p in b]), min([p[0] for p in b])], [max([p[1] for p in b]), max([p[0] for p in b])]]
                st.success("Área Pronta!")

# --- 4. MOTOR DE ANÁLISE ---
if st.session_state['roi'] and connected:
    st.caption(f"Analisando: {st.session_state['roi_name']}")
    m = geemap.Map()
    if st.session_state['map_bounds']: m.fit_bounds(st.session_state['map_bounds'])
    
    if st.button(f"⚡ PROCESSAR: {modo}"):
        with st.spinner("Baixando dados do Sentinel-2..."):
            roi = st.session_state['roi']
            
            # Seleciona bandas: B3(Green), B4(Red), B5(RedEdge), B8(NIR), B11(SWIR1)
            # Trocamos B12 por B11 para melhor MNDWI
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).select(['B3', 'B4', 'B5', 'B8', 'B11'])
            
            # Tenta pegar a imagem mais recente com menos de 20% de nuvens
            img_hoje = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).sort('system:time_start', False).first()
            
            # Pega a data da imagem para auditoria
            try:
                date_info = img_hoje.date().format('dd/MM/YYYY').getInfo()
                st.session_state['img_date'] = date_info
            except:
                st.session_state['img_date'] = "Data Indisponível"

            layer_name = "Resultado"
            vis_params = {}
            
            # --- 1. DESMATAMENTO ---
            if "Desmatamento" in modo:
                ref_hist = s2.filterDate('2023-01-01', '2024-12-31').median()
                ndvi_now = img_hoje.normalizedDifference(['B8','B4'])
                ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
                alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
                st.session_state['domo_map'] = alerta.updateMask(alerta.connectedPixelCount(30).gte(15)).clip(roi)
                vis_params = {'palette': ['red']}
                layer_name = "Supressão"

            # --- 2. ESPELHO D'ÁGUA (MNDWI) ---
            elif "MNDWI" in modo:
                # CORREÇÃO: Usando MNDWI (Green - SWIR1) / (Green + SWIR1)
                # Muito melhor para açudes barrentos que o NDWI comum
                mndwi = img_hoje.normalizedDifference(['B3', 'B11'])
                
                # CORREÇÃO: Baixei o limiar para -0.1 para pegar água turva
                water_mask = mndwi.gt(-0.1)
                st.session_state['domo_map'] = mndwi.updateMask(water_mask).clip(roi)
                
                vis_params = {'min': -0.1, 'max': 0.5, 'palette': ['white', 'blue', 'navy']}
                layer_name = "Corpos Hídricos"

            # --- 3. CLOROFILA (NDCI) ---
            elif "NDCI" in modo:
                # Usa o mesmo MNDWI robusto para achar a água primeiro
                mndwi = img_hoje.normalizedDifference(['B3', 'B11'])
                water_mask = mndwi.gt(-0.1)
                
                # Calcula NDCI apenas onde tem água
                ndci = img_hoje.normalizedDifference(['B5', 'B4']).updateMask(water_mask)
                st.session_state['domo_map'] = ndci.clip(roi)
                
                # Ajuste de visualização para realçar algas
                vis_params = {'min': 0.0, 'max': 0.15, 'palette': ['blue', 'cyan', 'lime', 'yellow', 'red']}
                layer_name = "Clorofila-a"

            # --- 4. QUEIMADAS (NBR) ---
            elif "NBR" in modo:
                nbr = img_hoje.normalizedDifference(['B8', 'B11'])
                st.session_state['domo_map'] = nbr.updateMask(nbr.lt(-0.1)).clip(roi)
                vis_params = {'min': -0.5, 'max': -0.1, 'palette': ['black', 'orange', 'red']}
                layer_name = "Queimada"

            st.session_state['legend_title'] = layer_name
            st.session_state['vis_params'] = vis_params

    # EXIBIÇÃO
    if st.session_state.get('img_date'):
        st.info(f"📅 Data da Imagem de Satélite: {st.session_state['img_date']}")

    if st.session_state['domo_map']:
        vis = st.session_state.get('vis_params', {})
        name = st.session_state.get('legend_title', 'Layer')
        m.addLayer(st.session_state['domo_map'], vis, name)
        
        if "MNDWI" in modo:
            m.add_colorbar(vis, label="Índice de Água (Branco=Raso/Barro, Azul=Fundo)")
        elif "NDCI" in modo:
             m.add_colorbar(vis, label="Risco de Eutrofização (Vermelho=Crítico)")

    m.to_streamlit(height=600)
