import streamlit as st
import pandas as pd
import json
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LC_FAISS

# ==============================================================================
# CONFIGURAÇÕES DE UI/UX E CONSTANTES DE CONTROLE
# ==============================================================================
st.set_page_config(page_title="Arena de Debates Parlamentares", page_icon="🏛️", layout="wide")

# CONSTANTE DE CONTROLE: Altere facilmente o limite de deputados aqui!
MAX_DEPUTADOS = 4
K_DOCUMENTOS = 5

# Estilização Customizada CSS para os Cards e Abas Modernas
st.markdown("""
    <style>
    .deputado-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
        text-align: center;
        margin-bottom: 12px;
    }
    .deputado-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .avatar-img {
        width: 85px;
        height: 85px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #edf2f7;
        margin: 0 auto 10px auto;
        display: block;
    }
    .partido-badge {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
        margin-top: 4px;
    }
    .uf-badge {
        background-color: #f1f5f9;
        color: #475569;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 4px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# CARREGAMENTO E TRATAMENTO DOS DADOS (CACHEADO)
# ==============================================================================
@st.cache_resource
def carregar_dados_e_modelos():
    faiss_path = "./faiss_index"
    deputados_csv_path = "./deputado_export_2026-06-12_095625.csv"
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = LC_FAISS.load_local(faiss_path, embeddings_model, allow_dangerous_deserialization=True)
    
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
            
    df_deputados = pd.DataFrame(lista_deputados).drop_duplicates(subset=['id'])
    return vector_store, embeddings_model, df_deputados

try:
    vector_store, embeddings_model, df_deputados = carregar_dados_e_modelos()
except Exception as e:
    st.error(f"Erro ao inicializar arquivos locais. Detalhes: {e}")
    st.stop()

# ==============================================================================
# INICIALIZAÇÃO DOS ESTADOS DA SESSÃO (SESSION STATE)
# ==============================================================================
if 'selecionados' not in st.session_state:
    st.session_state.selecionados = []
if 'tela_atual' not in st.session_state:
    st.session_state.tela_atual = "config"  # "config" ou "plenaria"
if 'executar_debate' not in st.session_state:
    st.session_state.executar_debate = False

# ==============================================================================
# PAINEL DE CONTROLE LATERAL (SIDEBAR)
# ==============================================================================
st.sidebar.title("🎮 Painel de Controle")

api_key = st.sidebar.text_input("Chave de API do Google Gemini", type="password")
if not api_key:
    st.info("💡 Digite sua Google API Key na barra lateral esquerda para ativar a Inteligência Artificial.")
    st.stop()

genai.configure(api_key=api_key)

st.sidebar.markdown("---")

# Seleção de modelos para cada Agente
st.sidebar.markdown("### 🤖 Configuração dos Modelos IA")
modelos_disponiveis = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-flash-lite", "gemini-3.5-flash"]

modelo_orq_nome = st.sidebar.selectbox("Agente Orquestrador:", modelos_disponiveis, index=0)
modelo_per_nome = st.sidebar.selectbox("Agentes Deputados:", modelos_disponiveis, index=2)
modelo_med_nome = st.sidebar.selectbox("Agente Mediador:", modelos_disponiveis, index=3)

modelo_orquestrador = genai.GenerativeModel(modelo_orq_nome)
modelo_persona      = genai.GenerativeModel(modelo_per_nome)
modelo_mediador     = genai.GenerativeModel(modelo_med_nome)

st.sidebar.markdown("---")

# Input da Pauta
nova_pauta = st.sidebar.text_area(
    "📝 Proposta / Projeto de Lei:",
    value="Proposta de Emenda à Constituição para redução da maioridade penal para 16 anos em casos de crimes hediondos.",
    height=100
)

# Monitor de selecionados na lateral
st.sidebar.markdown(f"### 👥 Mesa do Debate ({len(st.session_state.selecionados)} de {MAX_DEPUTADOS})")
if st.session_state.selecionados:
    for id_sel in st.session_state.selecionados:
        dep_info = df_deputados[df_deputados['id'] == id_sel].iloc[0]
        st.sidebar.caption(f"✅ {dep_info['nome']} ({dep_info['partido']})")
else:
    st.sidebar.info("Nenhum deputado selecionado.")

st.sidebar.markdown("---")

# Botões de Ação na barra lateral
col_btn_iniciar, col_btn_reset = st.sidebar.columns(2)

with col_btn_iniciar:
    if st.button("🔥 Iniciar", type="primary", use_container_width=True, disabled=len(st.session_state.selecionados) == 0):
        st.session_state.tela_atual = "plenaria"
        st.session_state.executar_debate = True
        st.rerun()

with col_btn_reset:
    if st.button("♻️ Reiniciar", type="secondary", use_container_width=True):
        st.session_state.selecionados = []
        st.session_state.tela_atual = "config"
        st.session_state.executar_debate = False
        st.rerun()


# ==============================================================================
# LÓGICA AUXILIAR DOS AGENTES COM SUPORTE A STREAMING
# ==============================================================================
def agente_orquestrador(pauta: str):
    prompt = f"Você é o Orquestrador Técnico da Câmara. Extraia termos de busca e palavras-chave separados por vírgula ideais para buscar discursos antigos pertinentes a esta pauta: {pauta}"
    return modelo_orquestrador.generate_content(prompt).text.strip()

class AgenteRepresentante:
    def __init__(self, id_deputado, nome, partido):
        self.id_deputado = str(id_deputado)
        self.nome = nome
        self.partido = partido

    def obter_contexto_rag(self, termos_busca):
        q_emb = embeddings_model.embed_query(termos_busca)
        docs_recuperados = vector_store.similarity_search_by_vector(
            embedding=q_emb, k=K_DOCUMENTOS, filter={"id_deputado": self.id_deputado}
        )
        return "".join([f"<discurso>\n{d.page_content}\n</discurso>\n\n" for d in docs_recuperados])

    def responder_a_pauta_stream(self, pauta, contexto):
        prompt_persona = f"""
        Você é o parlamentar {self.nome} do partido {self.partido}. 
        Simule seu pronunciamento oficial em 1ª pessoa na Tribuna da Câmara sobre a seguinte pauta: {pauta}.
        Baseie-se fortemente no seu histórico de discursos fornecido abaixo. Caso o histórico seja escasso ou não mencione diretamente o assunto, adote uma postura ideológica coerente com seu partido ({self.partido}).
        
        Histórico Real Recuperado: 
        {contexto if contexto else "Nenhum histórico específico encontrado."}
        """
        # Ativa o stream=True para responder letra por letra na tela
        return modelo_persona.generate_content(prompt_persona, stream=True)


# ==============================================================================
# CORPO PRINCIPAL E ALTERNÂNCIA DE TELAS DINÂMICAS (UX REAL-TIME)
# ==============================================================================
st.title("🏛️ Arena de Debates Parlamentares Inteligente")

# Abas falsas de navegação usando botões superiores (Garante troca instantânea)
col_tab1, col_tab2 = st.columns(2)
with col_tab1:
    if st.button("👥 1. Mesa e Configuração", use_container_width=True, type="primary" if st.session_state.tela_atual == "config" else "secondary"):
        st.session_state.tela_atual = "config"
        st.rerun()
with col_tab2:
    if st.button("🎬 2. Plenária / Resultado", use_container_width=True, type="primary" if st.session_state.tela_atual == "plenaria" else "secondary", disabled=len(st.session_state.selecionados) == 0):
        st.session_state.tela_atual = "plenaria"
        st.rerun()

st.markdown("---")

# --- TELA 1: CONFIGURAÇÃO E CARDS ---
if st.session_state.tela_atual == "config":
    col_filtro_nome, col_filtro_partido = st.columns([2, 1])

    with col_filtro_nome:
        busca_nome = st.text_input("🔍 Filtrar por nome:", "", placeholder="Digite o nome do parlamentar...")

    with col_filtro_partido:
        lista_partidos = ["Todos"] + sorted(df_deputados['partido'].unique().tolist())
        partido_selecionado = st.selectbox("🎯 Filtrar por partido:", lista_partidos)

    df_filtrado = df_deputados
    if busca_nome:
        df_filtrado = df_filtrado[df_filtrado['nome'].str.contains(busca_nome, case=False)]
    if partido_selecionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['partido'] == partido_selecionado]

    st.markdown("### 📋 Parlamentares Disponíveis")
    if df_filtrado.empty:
        st.warning("Nenhum deputado encontrado com os filtros aplicados.")
    else:
        cols = st.columns(4)
        for idx, row in df_filtrado.reset_index().iterrows():
            col_atual = cols[idx % 4]
            with col_atual:
                id_dep = row['id']
                is_sel = id_dep in st.session_state.selecionados
                
                border_color = "#3b82f6" if is_sel else "#e2e8f0"
                bg_color = "#eff6ff" if is_sel else "#ffffff"
                
                st.markdown(f"""
                    <div class="deputado-card" style="border-color: {border_color}; background-color: {bg_color};">
                        <img src="{row['foto']}" class="avatar-img" alt="Foto">
                        <div style="font-weight: 700; font-size: 15px; color: #1e293b; min-height: 44px; display: flex; align-items: center; justify-content: center;">
                            {row['nome']}
                        </div>
                        <span class="partido-badge">{row['partido']}</span>
                        <span class="uf-badge">{row['uf']}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                if is_sel:
                    if st.button("Remover", key=f"del_{id_dep}", type="secondary", use_container_width=True):
                        st.session_state.selecionados.remove(id_dep)
                        st.rerun()
                else:
                    limite_atingido = len(st.session_state.selecionados) >= MAX_DEPUTADOS
                    if st.button("Selecionar", key=f"add_{id_dep}", type="primary", disabled=limite_atingido, use_container_width=True):
                        st.session_state.selecionados.append(id_dep)
                        st.rerun()

# --- TELA 2: PLENÁRIA COM STREAMING PARALELO ---
else:
    st.subheader("📋 Pauta em Julgamento")
    st.info(f"_{nova_pauta}_")
    
    # Se o botão de iniciar foi disparado, executa o streaming ao vivo na tela
    if st.session_state.executar_debate:
        
        with st.spinner("🧠 Agente Orquestrador analisando conceitos da pauta..."):
            palavras_chave = agente_orquestrador(nova_pauta)
        st.markdown(f"ℹ️ **Tags RAG Extraídas:** `{palavras_chave}`")
        st.markdown("---")
        
        st.subheader("🎤 Pronunciamentos Oficiais (Em tempo real)")
        
        # Cria colunas lado a lado para renderizar o streaming dos deputados simultaneamente
        cols_discursos = st.columns(len(st.session_state.selecionados))
        dict_discursos_finais = {}
        
        # Loop para processar e exibir cada deputado escrevendo na tela aos poucos
        for i, id_dep in enumerate(st.session_state.selecionados):
            meta = df_deputados[df_deputados['id'] == id_dep].iloc[0]
            
            with cols_discursos[i]:
                # Desenha a foto e o cabeçalho do chat fixo
                st.chat_message("user", avatar=meta['foto']).markdown(f"**{meta['nome']} ({meta['partido']})**")
                
                # Cria um container vazio (placeholder) onde o texto vai pingar dinamicamente
                caixa_texto_dinamica = st.empty()
                
                agente = AgenteRepresentante(meta['id'], meta['nome'], meta['partido'])
                contexto_recuperado = agente.obter_contexto_rag(palavras_chave)
                
                # Invoca a resposta em modo streaming
                resposta_stream = agente.responder_a_pauta_stream(nova_pauta, contexto_recuperado)
                
                texto_acumulado = ""
                for chunk in resposta_stream:
                    if chunk.text:
                        texto_acumulado += chunk.text
                        # Atualiza a tela a cada palavra nova recebida do Gemini
                        caixa_texto_dinamica.markdown(f"> {texto_acumulado}")
                
                # Guarda o texto completo final para passar ao mediador
                dict_discursos_finais[f"{meta['nome']} ({meta['partido']})"] = texto_acumulado

        st.markdown("---")
        st.subheader("⚖️ Relatório de Mediação e Clima Político")
        
        caixa_mediador_dinamica = st.empty()
        historico_mediador = "".join([f"📌 [Pronunciamento de {dep}]:\n{txt}\n\n" for dep, txt in dict_discursos_finais.items()])
        prompt_med = f"Você é o Mediador da Câmara. Avalie o debate gerado sobre a pauta '{nova_pauta}'. Elabore um relatório contendo: 1) Pontos de convergência, 2) Conflitos ideológicos e 3) Clima Político geral.\n\nDebate:\n{historico_mediador}"
        
        # Streaming para a resposta do mediador final
        mediador_stream = modelo_mediador.generate_content(prompt_med, stream=True)
        texto_med_acumulado = ""
        for chunk in mediador_stream:
            if chunk.text:
                texto_med_acumulado += chunk.text
                caixa_mediador_dinamica.info(texto_med_acumulado)
        
        # Desliga a flag de execução para o debate ficar estático na tela caso mude de filtro
        st.session_state.executar_debate = False
    else:
        st.warning("💤 Sessão finalizada. Caso queira rodar uma nova simulação, retorne para a aba de Configuração.")