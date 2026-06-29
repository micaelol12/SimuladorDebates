import streamlit as st
import pandas as pd
import json
from transformers import pipeline

@st.cache_resource
def carregar_df_deputados():
    deputados_csv_path = "./deputado_export_2026-06-12_095625.csv"
    df_raw = pd.read_csv(deputados_csv_path)
    lista_deputados = []
    
    for _, row in df_raw.iterrows():
        try:
            dados_json = json.loads(row['json'])
            status = dados_json.get('ultimoStatus', {})
            lista_deputados.append({
                'id': str(row['id']),
                'nome': status.get('nome', dados_json.get('nomeCivil', 'Não Identificado')),
                'partido': status.get('siglaPartido', 'S/P'),
                'uf': status.get('siglaUf', '-'),
                'foto': status.get('urlFoto', 'https://www.camara.leg.br/internet/deputado/bandep/default.jpg')
            })
        except Exception:
            continue
            
    return pd.DataFrame(lista_deputados).drop_duplicates(subset=['id'])

@st.cache_resource
def carregar_analisador_sentimento():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def obter_cor_politica(texto: str):
    try:
        analisador = carregar_analisador_sentimento()
        resultado = analisador(texto[:512])[0]
        estrelas = int(resultado['label'].split()[0])
        
        if estrelas == 1: return "#fee2e2", "#991b1b", "Altamente Opositor"
        elif estrelas == 2: return "#ffedd5", "#c2410c", "Crítico"
        elif estrelas == 3: return "#f1f5f9", "#475569", "Neutro / Moderado"
        elif estrelas == 4: return "#ecfdf5", "#047857", "Favorável"
        else: return "#dcfce7", "#166534", "Altamente Defensor"
    except Exception:
        return "#f1f5f9", "#475569", "Não Analisado"