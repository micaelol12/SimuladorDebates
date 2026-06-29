import streamlit as st

MAX_DEPUTADOS = 4
K_DOCUMENTOS = 5
MIN_K_DOCUMENTOS = 2
MODELOS_DISPONIVEIS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-flash-lite", "gemini-3.5-flash"]

def configurar_pagina():
    st.set_page_config(page_title="Arena de Debates Parlamentares", page_icon="🏛️", layout="wide")
    st.markdown("""
        <style>
        .deputado-card {
            border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px;
            background-color: #ffffff; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease-in-out; text-align: center; margin-bottom: 12px;
        }
        .deputado-card:hover { transform: translateY(-4px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
        .avatar-img { width: 85px; height: 85px; border-radius: 50%; object-fit: cover; border: 3px solid #edf2f7; margin: 0 auto 10px auto; display: block; }
        .partido-badge { background-color: #e0f2fe; color: #0369a1; padding: 3px 12px; border-radius: 9999px; font-size: 12px; font-weight: 700; display: inline-block; margin-top: 4px; }
        .uf-badge { background-color: #f1f5f9; color: #475569; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 4px; display: inline-block; }
        .sentiment-box { padding: 10px; border-radius: 8px; margin-top: 10px; font-weight: bold; text-align: center; }
        </style>
    """, unsafe_allow_html=True)