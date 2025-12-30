import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import geopandas as gpd
import requests
import google.generativeai as genai
from shapely.geometry import shape
from shapely.ops import unary_union

# --- 1. CONFIGURAÇÃO DE SEGURANÇA (NUVEM) ---
st.set_page_config(layout="wide", page_title="DOMO v31.2 - Alpha Earth Ceará")

# Tenta ler das Secrets do Streamlit (Configurações avançadas no site)
if "API_KEY" in st.secrets:
    API_KEY = st.secrets["API_KEY"]
    PROJECT_ID = st.secrets["PROJECT_ID"]
else:
    # Fallback para teste local
    API_KEY = "AIzaSyDOfhla0Wv7ulRx-kOeBYO58Qfb8CFMzDY"
    PROJECT_ID = "domo-alpha-ia"

genai.configure(api_key=API_KEY)

# Inicialização Blindada para o Servidor
if 'ee_init' not in st.session_state:
    try:
        # Na nuvem, o Earth Engine usa as credenciais do Projeto registrado
        ee.Initialize(project=PROJECT_ID)
        st.session_state['ee_init'] = True
    except Exception as e:
        # Se falhar na nuvem, tenta o método de autenticação padrão
        try:
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID)
            st.session_state['ee_init'] = True
        except:
            st.error(f"Erro de Conexão: O projeto {PROJECT_ID} precisa estar registrado no Earth Engine.")

@st.cache_resource
def load_smart_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                return genai.GenerativeModel(m.name)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

model = load_smart_model()

# Inicialização de Variáveis de Estado
for key in ['roi', 'domo_map', 'last_scan_data', 'roi_name', 'map_bounds', 'report_text']:
    if key not in st.session_state: st.session_state[key] = None

# --- 2. FUNÇÕES AUXILIARES ---
def clean_date(d): return d.strftime('%Y-%m-%d')

@st.cache_data
def get_municipio_geometry(mun_id):
    url = f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{mun_id}?formato=application/vnd.geo+json&qualidade=intermediaria"
    try:
        data = requests.get(url).json()
        geoms = [shape(f['geometry']) for f in data['features']] if data.get('type') == 'FeatureCollection' else [shape(data.get('geometry', data))]
        return unary_union(geoms).buffer(0).simplify(0.0001)
    except: return None

# --- 3. SIDEBAR (FOCO CEARÁ) ---
with st.sidebar:
    st.title("🌵 DOMO - Semiárido")
    st.caption(f"Projeto Ativo: {PROJECT_ID}")
    
    if st.button("⚖️ AUDITORIA ALPHA (CEARÁ)", type="primary", use_container_width=True):
        if st.session_state.get('last_scan_data') and model:
            prompt = f"Analise o alerta: {st.session_state['last_scan_data']} no Ceará. É desmatamento ou seca sazonal?"
            try: st.info(model.generate_content(prompt).text)
            except: st.error("Erro na IA.")

    st.divider()
    try:
        estados = requests.get("https://servicodados.ibge.gov.br/api/v1/localidades/estados?orderBy=nome").json()
        uf = st.selectbox("Estado", [e['sigla'] for e in estados], index=5) # Ceará
        uf_id = [e['id'] for e in estados if e['sigla'] == uf][0]
        municipios = requests.get(f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf_id}/municipios?orderBy=nome").json()
        mun_nomes = st.multiselect("Municípios do Ceará", [m['nome'] for m in municipios])
    except: st.error("Erro IBGE")

    if st.button("📍 MAPEAR ÁREA"):
        if mun_nomes:
            with st.spinner("Buscando limites Geográficos..."):
                geom_list = [get_municipio_geometry([m['id'] for m in municipios if m['nome'] == n][0]) for n in mun_nomes]
                gdf = gpd.GeoDataFrame(geometry=[g for g in geom_list if g], crs="EPSG:4326")
                st.session_state['roi'] = geemap.geopandas_to_ee(gdf).geometry()
                st.session_state['roi_name'] = f"{', '.join(mun_nomes)} - CE"
                b = st.session_state['roi'].bounds().getInfo()['coordinates'][0]
                st.session_state['map_bounds'] = [[min([p[1] for p in b]), min([p[0] for p in b])], [max([p[1] for p in b]), max([p[0] for p in b])]]
                st.success("Área Carregada.")

# --- 4. MOTOR ALPHA: FILTRO DE CAATINGA ---
if st.session_state['roi']:
    tab_scan, tab_calor = st.tabs(["🛰️ MONITORAMENTO CEARÁ", "🔥 FOCOS DE CALOR"])

    with tab_scan:
        m1 = geemap.Map()
        if st.session_state['map_bounds']: m1.fit_bounds(st.session_state['map_bounds'])
        
        if st.button("⚡ ESCANEAR (FILTRO ANTI-SECA SAZONAL)"):
            with st.spinner("Analisando biomassa histórica..."):
                roi = st.session_state['roi']
                hoje = pd.Timestamp.now()
                
                s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
                img_hoje = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15)).sort('system:time_start', False).first()
                # Referência de 2 anos atrás para validar o que é perene
                ref_hist = s2.filterDate(clean_date(hoje - pd.Timedelta(days=730)), clean_date(hoje - pd.Timedelta(days=330))).median()
                
                if img_hoje.bandNames().contains('B8').getInfo():
                    ndvi_now = img_hoje.normalizedDifference(['B8','B4'])
                    ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
                    
                    # Filtro Sazonal: Solo hoje (<0.2) e Mata antes (>0.45)
                    alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
                    limpo = alerta.updateMask(alerta.connectedPixelCount(30).gte(10))
                    
                    area_ha = limpo.reduceRegion(reducer=ee.Reducer.count(), geometry=roi, scale=10, maxPixels=1e9).getInfo().get('nd', 0) * 0.01
                    
                    st.session_state['domo_map'] = limpo.clip(roi)
                    st.session_state['last_scan_data'] = f"Supressão de {area_ha:.2f} ha detectada no Ceará."
                    
                    if area_ha > 0.3: st.error(f"🚨 Alerta Real na Caatinga: {area_ha:.2f} ha")
                    else: st.success(f"✅ Área Estável ({area_ha:.2f} ha)")
                else: st.warning("Céu nublado. Tente outra data.")

        if st.session_state['domo_map']: m1.addLayer(st.session_state['domo_map'], {'palette': ['red']}, "Alerta Caatinga")
        m1.to_streamlit(height=500)
