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

# --- 1. CONFIGURAÇÃO DE AMBIENTE E SEGURANÇA ---
st.set_page_config(layout="wide", page_title="DOMO Alpha Earth - Ceará")

# Puxa credenciais das Secrets do Streamlit Cloud
PROJECT_ID = st.secrets.get("PROJECT_ID", "domo-alpha-ia")
API_KEY = st.secrets.get("API_KEY")
EE_KEYS_JSON = st.secrets.get("EE_KEYS")

# Configura IA Gemini
if API_KEY:
    genai.configure(api_key=API_KEY)

@st.cache_resource
def init_ee_enterprise():
    """Autenticação profissional via Service Account para evitar erros na nuvem."""
    try:
        if EE_KEYS_JSON:
            # Converte a string JSON das Secrets em dicionário
            key_dict = json.loads(EE_KEYS_JSON)
            credentials = ee.ServiceAccountCredentials(
                key_dict['client_email'], 
                key_data=EE_KEYS_JSON
            )
            ee.Initialize(credentials, project=PROJECT_ID)
            return True
        else:
            # Tenta inicialização padrão caso as chaves não existam (local)
            ee.Initialize(project=PROJECT_ID)
            return True
    except Exception as e:
        st.error(f"Erro Crítico de Conexão Google: {e}")
        return False

# Inicializa o motor do Google Earth Engine
connected = init_ee_enterprise()

@st.cache_resource
def load_model():
    """Carrega o modelo de IA para laudos técnicos."""
    try:
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

model = load_model()

# --- 2. FUNÇÕES DE ALTA PERFORMANCE (CACHE) ---
@st.cache_data(ttl=86400)
def get_ceara_cities():
    """Busca cidades do Ceará no IBGE e guarda em cache por 24h."""
    return requests.get("https://servicodados.ibge.gov.br/api/v1/localidades/estados/23/municipios?orderBy=nome").json()

@st.cache_data(ttl=86400)
def get_fast_geometry(mun_ids):
    """Obtém e simplifica a geometria para carregamento rápido no mapa."""
    geoms = []
    for m_id in mun_ids:
        url = f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{m_id}?formato=application/vnd.geo+json&qualidade=minima"
        try:
            data = requests.get(url).json()
            g = shape(data['features'][0]['geometry']) if 'features' in data else shape(data['geometry'])
            geoms.append(g.simplify(0.005)) 
        except: continue
    return unary_union(geoms).buffer(0)

def clean_date(d): return d.strftime('%Y-%m-%d')

# Variáveis de Sessão
for key in ['roi', 'domo_map', 'roi_name', 'map_bounds', 'last_scan_data']:
    if key not in st.session_state: st.session_state[key] = None

# --- 3. INTERFACE LATERAL (PAINEL DE CONTROLE) ---
with st.sidebar:
    st.title("🌵 DOMO Alpha Earth")
    st.caption(f"Ceará | Projeto: {PROJECT_ID}")
    
    if connected: st.success("🛰️ Conexão Cloud: Ativa")
    else: st.error("❌ Erro de Autenticação")

    st.divider()
    
    # Seleção de Municípios
    try:
        municipios = get_ceara_cities()
        nomes_lista = [m['nome'] for m in municipios]
        selecao = st.multiselect("Selecione os Municípios", nomes_lista)
    except: st.error("Erro ao carregar base do IBGE.")

    if st.button("📍 CARREGAR ÁREA", type="primary", use_container_width=True):
        if selecao:
            with st.spinner("Mapeando limites..."):
                ids = [m['id'] for m in municipios if m['nome'] in selecao]
                geom = get_fast_geometry(ids)
                st.session_state['roi'] = geemap.geopandas_to_ee(gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")).geometry()
                st.session_state['roi_name'] = ", ".join(selecao)
                b = st.session_state['roi'].bounds().getInfo()['coordinates'][0]
                st.session_state['map_bounds'] = [[min([p[1] for p in b]), min([p[0] for p in b])], [max([p[1] for p in b]), max([p[0] for p in b])]]
                st.success("Zona Registrada!")

    if st.button("⚖️ GERAR LAUDO IA", use_container_width=True):
        if st.session_state['last_scan_data'] and model:
            with st.spinner("IA analisando contexto jurídico..."):
                prompt = f"Analise como Auditor Ambiental: {st.session_state['last_scan_data']} no Ceará. Cite o Código Florestal."
                st.info(model.generate_content(prompt).text)

# --- 4. PAINEL PRINCIPAL E MOTOR DE VARREDURA ---
if st.session_state['roi'] and connected:
    m = geemap.Map()
    if st.session_state['map_bounds']: m.fit_bounds(st.session_state['map_bounds'])
    
    if st.button("⚡ EXECUTAR VARREDURA ALPHA (FILTRO CAATINGA)"):
        with st.spinner("Analisando biomassa e histórico sazonal..."):
            roi = st.session_state['roi']
            # Seleção de bandas específicas para reduzir latência
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).select(['B4', 'B8', 'B12'])
            
            img_hoje = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).sort('system:time_start', False).first()
            # Comparação com média de 2 anos para eliminar efeito da seca
            ref_hist = s2.filterDate('2023-01-01', '2024-12-31').median()
            
            ndvi_now = img_hoje.normalizedDifference(['B8','B4'])
            ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
            
            # Regra de Ouro: Solo hoje (<0.2) onde antes era mata estável (>0.45)
            alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
            
            # Filtro de Conectividade para remover ruído (limite 0.15 ha)
            limpo = alerta.updateMask(alerta.connectedPixelCount(30).gte(15))
            
            # Cálculo de Área Otimizado
            stats = limpo.reduceRegion(reducer=ee.Reducer.count(), geometry=roi, scale=10, maxPixels=1e9)
            area_ha = stats.getInfo().get('nd', 0) * 0.01
            
            st.session_state['domo_map'] = limpo.clip(roi)
            st.session_state['last_scan_data'] = f"Alerta de {area_ha:.2f} ha em {st.session_state['roi_name']}."
            
            if area_ha > 0.3:
                st.error(f"🚨 SUPRESSÃO DETECTADA: {area_ha:.2f} ha")
            else:
                st.success("✅ ÁREA ESTÁVEL: Sem supressão inédita detectada.")

    if st.session_state['domo_map']:
        m.addLayer(st.session_state['domo_map'], {'palette': ['red']}, "Alerta Alpha")
    
    m.to_streamlit(height=600)

else:
    st.info("👈 Por favor, carregue os municípios do Ceará no menu lateral para iniciar.")
