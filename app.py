import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import geopandas as gpd
import requests
import google.generativeai as genai
from shapely.geometry import shape
from shapely.ops import unary_union

# --- 1. SETUP E CONEXÃO CLOUD (ZERO INTERAÇÃO) ---
st.set_page_config(layout="wide", page_title="DOMO v32.2 - Cloud Fixed")

# Puxa credenciais das Secrets do Streamlit
API_KEY = st.secrets.get("API_KEY", "SUA_CHAVE_AQUI")
PROJECT_ID = st.secrets.get("PROJECT_ID", "domo-alpha-ia")

genai.configure(api_key=API_KEY)

@st.cache_resource
def init_ee_cloud():
    """Inicializa o Earth Engine usando o ID do projeto registrado."""
    try:
        # Tenta inicializar diretamente sem chamar Authenticate()
        ee.Initialize(project=PROJECT_ID)
        return True
    except Exception as e:
        # Exibe erro técnico caso o projeto não esteja vinculado no console GCP
        st.error(f"Erro de Conexão na Nuvem: {e}")
        return False

# Executa inicialização
connected = init_ee_cloud()

@st.cache_resource
def load_model():
    try:
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

model = load_model()

# --- 2. CACHE DE DADOS (VELOCIDADE NO CEARÁ) ---
@st.cache_data(ttl=86400)
def get_fast_geometry(mun_ids):
    geoms = []
    for m_id in mun_ids:
        url = f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{m_id}?formato=application/vnd.geo+json&qualidade=minima"
        try:
            data = requests.get(url).json()
            g = shape(data['features'][0]['geometry']) if 'features' in data else shape(data['geometry'])
            geoms.append(g.simplify(0.005)) # Reduz o peso do mapa
        except: continue
    return unary_union(geoms).buffer(0)

# Inicialização de Variáveis
for key in ['roi', 'domo_map', 'roi_name', 'map_bounds']:
    if key not in st.session_state: st.session_state[key] = None

# --- 3. SIDEBAR (FOCO CEARÁ) ---
with st.sidebar:
    st.title("🌵 DOMO - Semiárido")
    st.caption(f"Status: Cloud Online | Projeto: {PROJECT_ID}")
    
    @st.cache_data(ttl=86400)
    def get_ceara_cities():
        return requests.get("https://servicodados.ibge.gov.br/api/v1/localidades/estados/23/municipios?orderBy=nome").json()
    
    try:
        municipios = get_ceara_cities()
        mun_nomes = st.multiselect("Municípios do Ceará", [m['nome'] for m in municipios])
    except: st.error("Erro IBGE")

    if st.button("📍 CARREGAR MUNICÍPIO"):
        if mun_nomes:
            ids = [m['id'] for m in municipios if m['nome'] in mun_nomes]
            geom = get_fast_geometry(ids)
            st.session_state['roi'] = geemap.geopandas_to_ee(gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")).geometry()
            st.session_state['roi_name'] = ", ".join(mun_nomes)
            st.success("Mapa Carregado!")

# --- 4. MOTOR ALPHA: MONITORAMENTO ---
if st.session_state['roi'] and connected:
    m = geemap.Map()
    
    if st.button("⚡ ESCANEAR CAATINGA (NDVI+NBR)"):
        with st.spinner("Analisando biomassa via Satélite..."):
            roi = st.session_state['roi']
            # Seleciona apenas bandas essenciais para economizar memória do servidor
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).select(['B4', 'B8', 'B12'])
            
            img_hoje = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).sort('system:time_start', False).first()
            ref_hist = s2.filterDate('2023-01-01', '2024-12-31').median()
            
            # Filtro Sazonal: Solo hoje (<0.2) vs Floresta antes (>0.45)
            ndvi_now = img_hoje.normalizedDifference(['B8','B4'])
            ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
            
            alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
            limpo = alerta.updateMask(alerta.connectedPixelCount(30).gte(15)) # Filtro de conectividade
            
            st.session_state['domo_map'] = limpo.clip(roi)
            st.success("Varredura Concluída!")

    if st.session_state['domo_map']:
        m.addLayer(st.session_state['domo_map'], {'palette':['red']}, "Alerta")
    m.to_streamlit(height=500)
