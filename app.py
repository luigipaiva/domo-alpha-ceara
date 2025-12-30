import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import geopandas as gpd
import requests
import google.generativeai as genai
from shapely.geometry import shape
from shapely.ops import unary_union

# --- 1. SETUP E CACHE DE CONEXÃO ---
st.set_page_config(layout="wide", page_title="DOMO v32.0 - Alta Performance")

# Puxa credenciais das Secrets
API_KEY = st.secrets.get("API_KEY", "AIzaSyDOfhla0Wv7ulRx-kOeBYO58Qfb8CFMzDY")
PROJECT_ID = st.secrets.get("PROJECT_ID", "domo-alpha-ia")

genai.configure(api_key=API_KEY)

@st.cache_resource
def init_ee():
    """Inicializa o EE uma única vez e mantém a conexão 'quente'."""
    try:
        ee.Initialize(project=PROJECT_ID)
        return True
    except:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
        return True

init_ee()

@st.cache_resource
def load_model():
    return genai.GenerativeModel('gemini-1.5-flash')

model = load_model()

# --- 2. CACHE DE GEOMETRIA (ACELERA O MAPA) ---
@st.cache_data(ttl=86400) # Guarda por 24h para não baixar do IBGE toda hora
def get_fast_geometry(mun_ids):
    geoms = []
    for m_id in mun_ids:
        url = f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{m_id}?formato=application/vnd.geo+json&qualidade=minima"
        try:
            data = requests.get(url).json()
            g = shape(data['features'][0]['geometry']) if 'features' in data else shape(data['geometry'])
            # Simplifica a geometria para reduzir o peso do GeoJSON
            geoms.append(g.simplify(0.005, preserve_topology=True))
        except: continue
    union = unary_union(geoms).buffer(0)
    return union

# --- 3. INTERFACE OTIMIZADA ---
with st.sidebar:
    st.title("🌵 DOMO High Speed")
    st.caption(f"Status: Cloud Turbo | Projeto: {PROJECT_ID}")
    
    # Busca de Cidades (Cache de 24h para ser instantâneo)
    @st.cache_data(ttl=86400)
    def get_ceara_cities():
        return requests.get("https://servicodados.ibge.gov.br/api/v1/localidades/estados/23/municipios?orderBy=nome").json()
    
    try:
        municipios = get_ceara_cities()
        mun_nomes = st.multiselect("Selecione Cidades do Ceará", [m['nome'] for m in municipios])
    except: st.error("Erro ao carregar cidades.")

    if st.button("📍 CARREGAR IMEDIATO"):
        if mun_nomes:
            ids = [m['id'] for m in municipios if m['nome'] in mun_nomes]
            geom = get_fast_geometry(ids)
            st.session_state['roi'] = geemap.geopandas_to_ee(gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")).geometry()
            st.session_state['roi_name'] = ", ".join(mun_nomes)
            st.success("Mapa Pronto!")

# --- 4. MOTOR DE VARREDURA LEVE ---
if 'roi' in st.session_state:
    m = geemap.Map()
    
    if st.button("⚡ VARREDURA FLASH (CAATINGA)"):
        with st.spinner("Análise em tempo real..."):
            roi = st.session_state['roi']
            # Filtra a coleção APENAS para as bandas necessárias (B4, B8, B12)
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).select(['B4', 'B8', 'B12'])
            
            img_agora = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).sort('system:time_start', False).first()
            ref_hist = s2.filterDate('2023-01-01', '2024-12-31').median()
            
            ndvi_now = img_agora.normalizedDifference(['B8','B4'])
            ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
            
            # Filtro Sazonal Ceará
            alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
            limpo = alerta.updateMask(alerta.connectedPixelCount(30).gte(15))
            
            # ReduceRegion com escala maior para rapidez inicial
            area = limpo.reduceRegion(reducer=ee.Reducer.count(), geometry=roi, scale=20, maxPixels=1e9).getInfo().get('nd', 0) * 0.04
            
            st.session_state['domo_map'] = limpo.clip(roi)
            st.error(f"Alerta: {area:.2f} ha") if area > 0.3 else st.success("Área Estável")

    if st.session_state.get('domo_map'): m.addLayer(st.session_state['domo_map'], {'palette':['red']}, "Supressão")
    m.to_streamlit(height=500)
