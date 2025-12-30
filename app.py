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
import datetime

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(layout="wide", page_title="DOMO Monitor")

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

# --- 2. FUNÇÕES AUXILIARES E CACHE ---

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
    """Cálculo otimizado de área estática"""
    pixel_area = ee.Image.pixelArea().updateMask(image_mask)
    stats = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e10,
        bestEffort=True
    )
    area_m2 = stats.getInfo().get('area', 0)
    return area_m2 / 10000

def get_time_series_chart(collection, geometry, metric_band, reducer=ee.Reducer.mean(), scale=100):
    """Gera dados para gráfico de série temporal (NOVO)"""
    def extract_stats(img):
        stats = img.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        )
        return img.set('stats', stats)

    # Mapeia a redução sobre a coleção (limitada aos ultimos 20 elementos para performance)
    col_w_stats = collection.limit(20).map(extract_stats)
    stats_list = col_w_stats.aggregate_array('stats').getInfo()
    dates_list = col_w_stats.aggregate_array('system:time_start').getInfo()
    
    data = []
    for i, date_ms in enumerate(dates_list):
        if stats_list[i] and metric_band in stats_list[i]:
            val = stats_list[i][metric_band]
            if val is not None:
                dt = datetime.datetime.fromtimestamp(date_ms / 1000.0)
                data.append({'Data': dt, 'Valor': val})
    
    return pd.DataFrame(data)

# Inicializa Session State
keys_to_init = ['roi', 'domo_map', 'roi_name', 'map_bounds', 'legend_title', 'sat_source', 'map_key', 'metric_area', 'chart_data', 'download_url']
for key in keys_to_init:
    if key not in st.session_state: st.session_state[key] = None

# --- 3. SIDEBAR (INTERFACE) ---
with st.sidebar:
    st.title("⚡ DOMO Turbo")
    st.markdown("---")
    
    modo = st.radio(
        "🎯 Objetivo do Monitoramento:",
        ["🌳 Desmatamento (Sentinel)", "💧 Espelho D'água (Landsat)", "🧪 Clorofila (Sentinel)", "🔥 Queimadas (Sentinel)"]
    )
    
    st.markdown("### 📅 Período")
    data_fim = st.date_input("Data de Análise", datetime.date.today())
    # O inicio é calculado automaticamente para performance no histórico
    
    st.markdown("---")
    
    try:
        municipios = get_ceara_cities()
        selecao = st.multiselect("📍 Municípios (CE)", [m['nome'] for m in municipios])
    except: st.error("Erro IBGE")

    if st.button("CARREGAR ÁREA", type="primary", disabled=not connected):
        if selecao:
            with st.spinner("Construindo geometria..."):
                st.session_state['domo_map'] = None
                st.session_state['chart_data'] = None
                st.session_state['download_url'] = None
                
                ids = [m['id'] for m in municipios if m['nome'] in selecao]
                geom = get_fast_geometry(ids)
                
                if geom:
                    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
                    st.session_state['roi'] = geemap.geopandas_to_ee(gdf).geometry()
                    st.session_state['roi_name'] = ", ".join(selecao)
                    b = st.session_state['roi'].bounds().getInfo()['coordinates'][0]
                    st.session_state['map_bounds'] = [[min([p[1] for p in b]), min([p[0] for p in b])], [max([p[1] for p in b]), max([p[0] for p in b])]]
                    st.session_state['map_key'] = str(uuid.uuid4())
                    st.success(f"Alvo Definido: {st.session_state['roi_name']}")

# --- 4. MOTOR DE PROCESSAMENTO ---
if st.session_state['roi'] and connected:
    
    # Header de Métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("Local", f"{len(selecao)} Município(s)")
    c2.metric("Área Detectada", f"{st.session_state['metric_area']:.2f} ha" if st.session_state['metric_area'] is not None else "--")
    c3.metric("Fonte", st.session_state['sat_source'] if st.session_state['sat_source'] else "--")

    # Botão de Execução Principal
    if st.button(f"🚀 EXECUTAR ANÁLISE: {modo.split('(')[0]}", use_container_width=True):
        roi = st.session_state['roi']
        # Datas para o Earth Engine
        ee_date = ee.Date(data_fim.strftime('%Y-%m-%d'))
        
        vis_params = {}
        layer_name = ""
        mask_final = None
        img_final = None # Para exportação
        scale_calc = 30
        col_for_chart = None
        band_for_chart = None
        
        # --- LÓGICA LANDSAT (ÁGUA) ---
        if "Landsat" in modo:
            with st.spinner("Processando Landsat 9..."):
                l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(ee_date.advance(-2, 'month'), ee_date)
                img = l9.filter(ee.Filter.lt('CLOUD_COVER', 20)).sort('system:time_start', False).first()
                
                if img:
                    green = img.select('SR_B3').multiply(0.0000275).add(-0.2)
                    swir = img.select('SR_B6').multiply(0.0000275).add(-0.2)
                    mndwi = green.subtract(swir).divide(green.add(swir)).rename('MNDWI')
                    mask_final = mndwi.gt(0.0)
                    
                    st.session_state['domo_map'] = mndwi.updateMask(mask_final).clip(roi)
                    st.session_state['sat_source'] = "Landsat 9"
                    vis_params = {'min': 0, 'max': 0.6, 'palette': ['white', 'blue', 'navy']}
                    layer_name = "Água"
                    img_final = mndwi
                    
                    # Setup para o gráfico
                    col_for_chart = l9.select('SR_B3').map(lambda i: i.set('system:time_start', i.get('system:time_start'))) # Simplificação para o exemplo
                    # (Idealmente recalcularíamos MNDWI para toda coleção, mas é pesado para demo)
                else: st.warning("Sem imagem Landsat limpa no período.")

        # --- LÓGICA SENTINEL (GERAL) ---
        else:
            with st.spinner("Processando Sentinel-2..."):
                s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
                # Pega imagem mais recente próxima à data escolhida
                img = s2.filterDate(ee_date.advance(-1, 'month'), ee_date)\
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
                        .sort('system:time_start', False).first()
                
                if img:
                    st.session_state['sat_source'] = "Sentinel-2"
                    
                    if "Desmatamento" in modo:
                        # Comparação histórica OTIMIZADA
                        start_hist = ee_date.advance(-13, 'month')
                        end_hist = ee_date.advance(-11, 'month')
                        
                        ref_hist = s2.filterDate(start_hist, end_hist).median()
                        ndvi_now = img.normalizedDifference(['B8','B4']).rename('NDVI')
                        ndvi_ref = ref_hist.normalizedDifference(['B8','B4'])
                        
                        alerta = ndvi_now.lt(0.2).And(ndvi_ref.gt(0.45)).selfMask()
                        mask_final = alerta.connectedPixelCount(30).gte(15)
                        
                        st.session_state['domo_map'] = alerta.updateMask(mask_final).clip(roi)
                        vis_params = {'palette': ['red']}
                        layer_name = "Supressão (Desmatamento)"
                        scale_calc = 20
                        img_final = alerta
                        
                        # Gráfico: Variação do NDVI médio na área
                        col_for_chart = s2.filterDate(ee_date.advance(-6, 'month'), ee_date)\
                                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
                                          .map(lambda i: i.addBands(i.normalizedDifference(['B8','B4']).rename('NDVI')))
                        band_for_chart = 'NDVI'

                    elif "Clorofila" in modo:
                        scale_calc = 20
                        water_mask = img.normalizedDifference(['B3', 'B11']).gt(-0.1)
                        ndci = img.normalizedDifference(['B5', 'B4']).rename('NDCI')
                        mask_final = water_mask
                        
                        st.session_state['domo_map'] = ndci.updateMask(water_mask).clip(roi)
                        vis_params = {'min': 0, 'max': 0.15, 'palette': ['blue', 'lime', 'red']}
                        layer_name = "Índice de Clorofila"
                        img_final = ndci
                        
                        col_for_chart = s2.filterDate(ee_date.advance(-3, 'month'), ee_date)\
                                          .map(lambda i: i.addBands(i.normalizedDifference(['B5', 'B4']).rename('NDCI')))
                        band_for_chart = 'NDCI'

                    elif "Queimadas" in modo:
                        scale_calc = 20
                        nbr = img.normalizedDifference(['B8', 'B12']).rename('NBR')
                        mask_final = nbr.lt(-0.1)
                        
                        st.session_state['domo_map'] = nbr.updateMask(mask_final).clip(roi)
                        vis_params = {'min': -0.5, 'max': -0.1, 'palette': ['black', 'orange', 'red']}
                        layer_name = "Cicatriz de Fogo"
                        img_final = nbr
                        
                        col_for_chart = s2.filterDate(ee_date.advance(-3, 'month'), ee_date)\
                                          .map(lambda i: i.addBands(i.normalizedDifference(['B8', 'B12']).rename('NBR')))
                        band_for_chart = 'NBR'

                else: st.warning(f"Sem imagem Sentinel limpa perto de {data_fim}.")

        # FINALIZAÇÃO DO PROCESSAMENTO
        if mask_final is not None:
            # 1. Calcula Área
            area_ha = calculate_hectares(mask_final, roi, scale_calc)
            st.session_state['metric_area'] = area_ha
            st.session_state['legend_title'] = layer_name
            st.session_state['vis_params'] = vis_params
            
            # 2. Gera Dados do Gráfico (Se aplicável)
            if col_for_chart and band_for_chart:
                with st.spinner("Gerando estatísticas temporais..."):
                    df = get_time_series_chart(col_for_chart, roi, band_for_chart)
                    st.session_state['chart_data'] = df
            
            # 3. Gera URL de Download (Exportação Rápida)
            try:
                # Gera URL para GeoTIFF da área visualizada
                url = img_final.clip(roi).getDownloadURL({
                    'scale': 100, # Escala reduzida para download rápido via URL
                    'crs': 'EPSG:4326',
                    'region': roi
                })
                st.session_state['download_url'] = url
            except:
                st.session_state['download_url'] = None

            st.rerun()

    # --- 5. VISUALIZAÇÃO E RESULTADOS ---
    
    # A) MAPA
    m = geemap.Map()
    if st.session_state['map_bounds']: m.fit_bounds(st.session_state['map_bounds'])
    
    if st.session_state['domo_map']:
        vis = st.session_state.get('vis_params', {})
        name = st.session_state.get('legend_title', 'Result')
        m.addLayer(st.session_state['domo_map'], vis, name)
        
        # Legendas Dinâmicas
        if "Desmatamento" in modo: m.add_legend(title="Alerta", labels=["Desmatamento"], colors=["#FF0000"])
        elif "Landsat" in modo: m.add_colorbar(vis, label="NDWI (Água)")
        elif "Clorofila" in modo: m.add_colorbar(vis, label="NDCI (Algas)")
        elif "Queimadas" in modo: m.add_colorbar(vis, label="NBR (Fogo)")

    key_mapa = st.session_state.get('map_key', 'map_default')
    m.to_streamlit(height=500, key=key_mapa)

    # B) DASHBOARD DE DADOS (GRÁFICOS E EXPORTAÇÃO)
    if st.session_state.get('chart_data') is not None or st.session_state.get('metric_area') is not None:
        st.markdown("### 📊 Monitoramento e Exportação")
        
        tab1, tab2 = st.tabs(["📈 Evolução Temporal", "📥 Exportar Dados"])
        
        with tab1:
            df = st.session_state['chart_data']
            if df is not None and not df.empty:
                st.line_chart(df, x='Data', y='Valor', color="#FF4B4B")
                st.caption(f"Variação média do índice ({modo.split('(')[0]}) nos últimos meses.")
            else:
                st.info("Gráfico não disponível ou dados insuficientes para esta área.")
        
        with tab2:
            col_ex1, col_ex2 = st.columns(2)
            
            with col_ex1:
                st.markdown("**Relatório CSV**")
                if df is not None and not df.empty:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📄 Baixar Planilha (.csv)",
                        data=csv,
                        file_name=f"domo_dados_{st.session_state['roi_name']}.csv",
                        mime='text/csv',
                    )
                else: st.write("Gere o gráfico primeiro para baixar CSV.")
            
            with col_ex2:
                st.markdown("**Imagem GeoTIFF**")
                url_img = st.session_state.get('download_url')
                if url_img:
                    st.markdown(f"[⬇️ Baixar Imagem (GeoTIFF)]({url_img})", unsafe_allow_html=True)
                    st.caption("*Link válido por tempo limitado. Resolução ajustada (100m) para download rápido.*")
                else:
                    st.write("Processe a análise primeiro.")
