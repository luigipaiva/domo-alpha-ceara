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

# --- 1. CONFIGURAÇÃO E AUTENTICAÇÃO ---
st.set_page_config(layout="wide", page_title="DOMO Alpha Earth - Ceará")

PROJECT_ID = st.secrets.get("PROJECT_ID", "domo-alpha-ia")
API_KEY = st.secrets.get("API_KEY")
EE_KEYS_RAW = st.secrets.get("EE_KEYS")

if API_KEY:
    genai.configure(api_key=API_KEY)

@st.cache_resource
def init_ee_enterprise():
    """Autenticação Enterprise robusta."""
    try:
        if EE_KEYS_RAW:
            # Carrega o JSON permitindo alguns erros de controle (strict=False)
            # Isso resolve o erro "Invalid \escape" na leitura inicial
            key_dict = json.loads(EE_KEYS_RAW, strict=False)

            # CORREÇÃO DA CHAVE PRIVADA (O Pulo do Gato)
            # O Google exige quebras de linha REAIS (\n), mas o JSON traz literais (\\n)
            # Aqui forçamos essa conversão apenas no campo da senha
            if 'private_key' in key_dict:
                key_dict['private_key'] = key_dict['private_key'].replace('\\n', '\n')

            # Cria as credenciais com o dicionário corrigido
            credentials = ee.ServiceAccountCredentials(
                key_dict['client_email'], 
                key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials, project=PROJECT_ID)
            return True
        else:
            # Fallback para execução local
            ee.Initialize(project=PROJECT_ID)
            return True
    except Exception as e:
        st.error(f"❌ Erro de Autenticação: {e}")
        return False

# Executa a conexão
connected = init_ee_enterprise()

@st.cache_resource
def load_model():
    try: return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

model = load_model()

# --- 2. DADOS E GEOMETRIA (CACHE) ---
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
            if 'features' in data:
                g = shape(data['features'][0]['geometry'])
            else:
                g = shape(data['geometry'])
            geoms.append(g.simplify(0.005))
        except: continue
    return unary_union(geoms).buffer(0)

# Variáveis de Sessão
for key in ['roi', 'domo_map', 'roi_name', 'map_bounds', 'last_scan_data']:
    if key not in st.session_state: st.session_state[key] = None

# --- 3. INTERFACE ---
with st.sidebar:
    st.title("🌵 DOMO Alpha Earth")
    if connected: st.success("🛰️ Conexão Google: OK")
    else: st.error("⚠️ Falha na Conexão")

    st.divider()
    try:
        municipios = get_ceara_cities()
        selecao = st.multiselect("Cidades do Ceará", [m['nome'] for m in municipios])
    except: st.error("Erro IBGE")

    # Botão só ativa se estiver conectado
    if st.button("📍 CARREGAR ÁREA", type="primary", use_container_width=True, disabled=not connected):
        if selecao:
            with st.spinner("Carregando mapa..."):
                ids = [m['id'] for m in municipios if m['nome'] in selecao]
                geom = get_fast_geometry(ids)
                
                gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
                st.session_state['roi'] = geemap.geopandas_to_ee(gdf).geometry()
                
                st.session_state['roi_name'] = ", ".join(selecao)
                b = st.session_state['roi'].bounds().getInfo()['coordinates'][0]
                st.session_state['map_bounds'] = [[min([p[1] for p in b]), min([p[0] for p in b])], [max([p[1] for p in b]), max([p[0] for p in b])]]
                st.success("Área pronta!")

# --- 4. MOTOR DE VARREDURA ---
if st.session_state['roi'] and connected:
    m = geemap.Map()
    if st.session_state['map_bounds']: m.fit_bounds(st.session_state['map_bounds'])
    
    if st.button("⚡ ESCANEAR CAATINGA"):
        with st.spinner("Processando imagens de satélite..."):
            roi = st.session_state['roi']
            
            # Filtro de bandas para otimizar velocidade
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).select(['B4', 'B8', 'B12'])
            img_hoje = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).sort('system:time_start', False).first()
            ref_hist = s2.filterDate('2023-01-01', '2024-12-31').median()
            
            ndvi_now = img_hoje.normalizedDifference(['B8','B4'])
            ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
            
            # Algoritmo de Supressão: Queda de vigor em área historicamente densa
            alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
            limpo = alerta.updateMask(alerta.connectedPixelCount(30).gte(15))
            
            stats = limpo.reduceRegion(reducer=ee.Reducer.count(), geometry=roi, scale=10, maxPixels=1e9)
            area_ha = stats.getInfo().get('nd', 0) * 0.01
            
            st.session_state['domo_map'] = limpo.clip(roi)
            st.session_state['last_scan_data'] = f"Alerta de {area_ha:.2f} ha em {st.session_state['roi_name']}"
            
            if area_ha > 0.3: st.error(f"🚨 SUPRESSÃO DETECTADA: {area_ha:.2f} ha")
            else: st.success("✅ ÁREA ESTÁVEL")

    if st.session_state['domo_map']:
        m.addLayer(st.session_state['domo_map'], {'palette': ['red']}, "Alerta")
    
    m.to_streamlit(height=600)
