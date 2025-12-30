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
st.set_page_config(layout="wide", page_title="DOMO Alpha Earth - MultiSpectral")

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

for key in ['roi', 'domo_map', 'roi_name', 'map_bounds', 'last_scan_data', 'legend_title']:
    if key not in st.session_state: st.session_state[key] = None

# --- 3. INTERFACE LATERAL ---
with st.sidebar:
    st.title("🌵 DOMO Alpha Earth")
    if connected: st.success("🛰️ Sistema Online")
    else: st.error("⚠️ Sistema Offline")

    st.divider()
    
    # SELETOR DE MODO DE OPERAÇÃO
    modo = st.radio(
        "Selecione a Lente de Análise:",
        ["🌳 Desmatamento (Caatinga)", "💧 Espelho D'água (NDWI)", "🧪 Qualidade da Água (NDCI)", "🔥 Cicatrizes de Fogo (NBR)"]
    )
    
    st.divider()
    try:
        municipios = get_ceara_cities()
        selecao = st.multiselect("Municípios do Ceará", [m['nome'] for m in municipios])
    except: st.error("Erro IBGE")

    if st.button("📍 CARREGAR ÁREA", type="primary", use_container_width=True, disabled=not connected):
        if selecao:
            with st.spinner("Configurando satélite..."):
                ids = [m['id'] for m in municipios if m['nome'] in selecao]
                geom = get_fast_geometry(ids)
                gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
                st.session_state['roi'] = geemap.geopandas_to_ee(gdf).geometry()
                st.session_state['roi_name'] = ", ".join(selecao)
                b = st.session_state['roi'].bounds().getInfo()['coordinates'][0]
                st.session_state['map_bounds'] = [[min([p[1] for p in b]), min([p[0] for p in b])], [max([p[1] for p in b]), max([p[0] for p in b])]]
                st.success("Área Pronta!")

# --- 4. MOTOR MULTI-ESPECTRAL ---
if st.session_state['roi'] and connected:
    m = geemap.Map()
    if st.session_state['map_bounds']: m.fit_bounds(st.session_state['map_bounds'])
    
    if st.button(f"⚡ ESCANEAR: {modo.split('(')[0]}"):
        with st.spinner(f"Processando índice {modo.split('(')[-1][:-1]}..."):
            roi = st.session_state['roi']
            
            # Carrega coleção Sentinel-2 com TODAS as bandas necessárias
            # B3(Green), B4(Red), B5(RedEdge), B8(NIR), B12(SWIR)
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).select(['B3', 'B4', 'B5', 'B8', 'B12'])
            img_hoje = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).sort('system:time_start', False).first()
            
            layer_name = "Resultado"
            vis_params = {}
            
            # --- LÓGICA 1: DESMATAMENTO (NDVI) ---
            if "Desmatamento" in modo:
                ref_hist = s2.filterDate('2023-01-01', '2024-12-31').median()
                ndvi_now = img_hoje.normalizedDifference(['B8','B4'])
                ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
                alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
                resultado = alerta.updateMask(alerta.connectedPixelCount(30).gte(15))
                vis_params = {'palette': ['red']}
                layer_name = "Supressão Vegetal"

            # --- LÓGICA 2: ESPELHO D'ÁGUA (NDWI) ---
            elif "NDWI" in modo:
                # NDWI = (Green - NIR) / (Green + NIR) [McFeeters]
                ndwi = img_hoje.normalizedDifference(['B3', 'B8'])
                # Máscara: Apenas água (> 0.0)
                resultado = ndwi.updateMask(ndwi.gt(0.0))
                vis_params = {'min': 0, 'max': 0.5, 'palette': ['white', 'blue', 'navy']}
                layer_name = "Corpos Hídricos"

            # --- LÓGICA 3: CLOROFILA-A (NDCI) ---
            elif "NDCI" in modo:
                # Primeiro isolamos a água usando NDWI
                ndwi = img_hoje.normalizedDifference(['B3', 'B8'])
                water_mask = ndwi.gt(0.0)
                
                # NDCI = (RedEdge1 - Red) / (RedEdge1 + Red) [Proxy Sentinel-2]
                ndci = img_hoje.normalizedDifference(['B5', 'B4']).updateMask(water_mask)
                
                # Visualização: Azul (Limpa) -> Verde/Amarelo (Algas) -> Vermelho (Crítico)
                resultado = ndci
                vis_params = {'min': -0.1, 'max': 0.5, 'palette': ['blue', 'cyan', 'lime', 'yellow', 'red']}
                layer_name = "Índice de Clorofila"

            # --- LÓGICA 4: QUEIMADAS (NBR) ---
            elif "NBR" in modo:
                # NBR = (NIR - SWIR) / (NIR + SWIR)
                # Queimada recente tem NBR muito baixo ou negativo
                nbr = img_hoje.normalizedDifference(['B8', 'B12'])
                
                # Detecta cicatrizes severas (NBR < -0.1)
                resultado = nbr.updateMask(nbr.lt(-0.1))
                vis_params = {'min': -0.5, 'max': -0.1, 'palette': ['black', 'orange', 'red']}
                layer_name = "Cicatrizes de Fogo"

            # Processamento final e exibição
            if resultado:
                st.session_state['domo_map'] = resultado.clip(roi)
                st.session_state['legend_title'] = layer_name
                st.session_state['vis_params'] = vis_params
                st.success(f"Análise de {layer_name} concluída!")
            else:
                st.warning("Não foi possível gerar a imagem (nuvens ou dados indisponíveis).")

    # Adiciona a camada ao mapa se existir
    if st.session_state['domo_map']:
        vis = st.session_state.get('vis_params', {})
        name = st.session_state.get('legend_title', 'Layer')
        m.addLayer(st.session_state['domo_map'], vis, name)
        
        # Adiciona legenda explicativa dinâmica
        if "Desmatamento" in modo:
            m.add_legend(title="Legenda", labels=["Desmatamento Detectado"], colors=["#FF0000"])
        elif "NDWI" in modo:
            m.add_colorbar(vis, label="Índice de Água (Azul Escuro = Profundo)")
        elif "NDCI" in modo:
             m.add_colorbar(vis, label="Concentração de Algas (Vermelho = Crítico)")
        elif "NBR" in modo:
             m.add_legend(title="Severidade", labels=["Queimada Recente"], colors=["#FF4500"])

    m.to_streamlit(height=650)
