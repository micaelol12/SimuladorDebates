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

# Estilização Customizada CSS para os Cards
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
if 'aba_ativa' not in st.session_state:
    st.session_state.aba_ativa = 0
if 'historico_debate' not in st.session_state:
    st.session_state.historico_debate = None

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

# Melhoria 3: Seleção dinâmica de modelos para cada Agente
st.sidebar.markdown("### 🤖 Configuração dos Modelos IA")
modelos_disponiveis = [
    "gemini-2.5-flash", 
    "gemini-2.5-pro", 
    "gemini-3.1-flash-lite", 
    "gemini-3.5-flash"
]

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

# Ações principais estruturadas lado a lado para otimizar espaço
col_btn_iniciar, col_btn_reset = st.sidebar.columns(2)

with col_btn_iniciar:
    botao_debate = st.button(
        "🔥 Iniciar", 
        type="primary", 
        use_container_width=True, 
        disabled=len(st.session_state.selecionados) == 0
    )

with col_btn_reset:
    # Melhoria 2: Botão de reiniciar completo
    if st.button("♻️ Reiniciar", type="secondary", use_container_width=True):
        st.session_state.selecionados = []
        st.session_state.historico_debate = None
        st.session_state.aba_ativa = 0
        st.rerun()

# ==============================================================================
# LÓGICA DOS AGENTES COMPORTAMENTAIS
# ==============================================================================
def agente_orquestrador(pauta: str):
    prompt = f"Você é o Orquestrador Técnico da Câmara. Extraia termos de busca e palavras-chave separados por vírgula ideais para buscar discursos antigos pertinentes a esta pauta: {pauta}"
    return modelo_orquestrador.generate_content(prompt).text.strip()

class AgenteRepresentante:
    def __init__(self, id_deputado, nome, partido):
        self.id_deputado = str(id_deputado)
        self.nome = nome
        self.partido = partido

    def responder_a_pauta(self, pauta, termos_busca):
        q_emb = embeddings_model.embed_query(termos_busca)
        docs_recuperados = vector_store.similarity_search_by_vector(
            embedding=q_emb, k=3, filter={"id_deputado": self.id_deputado}
        )
        contexto = "".join([f"<discurso>\n{d.page_content}\n</discurso>\n\n" for d in docs_recuperados])
        
        prompt_persona = f"""
        Você é o parlamentar {self.nome} do partido {self.partido}. 
        Simule seu pronunciamento oficial em 1ª pessoa na Tribuna da Câmara sobre a seguinte pauta: {pauta}.
        Baseie-se fortemente no seu histórico de discursos fornecido abaixo. Caso o histórico seja escasso ou não mencione diretamente o assunto, adote uma postura ideológica coerente com seu partido ({self.partido}).
        
        Histórico Real Recuperado: 
        {contexto if contexto else "Nenhum histórico específico encontrado."}
        """
        return modelo_persona.generate_content(prompt_persona).text.strip()

def agente_mediador(pauta, rodada_debates):
    historico = "".join([f"📌 [Pronunciamento de {dep}]:\n{txt}\n\n" for dep, txt in rodada_debates.items()])
    prompt = f"Você é o Mediador da Câmara. Avalie o debate gerado sobre a pauta '{pauta}'. Elabore um relatório contendo: 1) Pontos de convergência, 2) Conflitos ideológicos e 3) Clima Político geral.\n\nDebate:\n{historico}"
    return modelo_mediador.generate_content(prompt).text.strip()

# ==============================================================================
# DISPARO DO FLUXO MULTIAGENTE
# ==============================================================================
if botao_debate:
    # Altera o estado para renderizar diretamente na aba de resultados
    st.session_state.aba_ativa = 1
    
    # Executa a simulação e salva os dados no estado para evitar perda em re-renderizações
    with st.spinner("🧠 Agente Orquestrador analisando conceitos da pauta..."):
        palavras_chave = agente_orquestrador(nova_pauta)
        
    debates_coletados = {}
    
    # Captura os discursos
    for id_dep in st.session_state.selecionados:
        meta = df_deputados[df_deputados['id'] == id_dep].iloc[0]
        agente = AgenteRepresentante(meta['id'], meta['nome'], meta['partido'])
        discurso_gerado = agente.responder_a_pauta(nova_pauta, palavras_chave)
        debates_coletados[id_dep] = {
            "nome": meta['nome'],
            "partido": meta['partido'],
            "foto": meta['foto'],
            "discurso": discurso_gerado
        }
                
    with st.spinner("⚖️ Agente Mediador ponderando argumentos..."):
        # Transforma o formato para passar ao mediador
        dict_mediador = {f"{v['nome']} ({v['partido']})": v['discurso'] for k, v in debates_coletados.items()}
        analise_final = agente_mediador(nova_pauta, dict_mediador)
        
    # Salva o pacote completo do resultado na sessão
    st.session_state.historico_debate = {
        "pauta": nova_pauta,
        "palavras_chave": palavras_chave,
        "discursos": debates_coletados,
        "mediacao": analise_final
    }

# ==============================================================================
# CORPO PRINCIPAL ORGANIZADO EM ABAS (UI/UX)
# ==============================================================================
st.title("🏛️ Arena de Debates Parlamentares Inteligente")

# Melhoria 1: Criação de abas para separar a seleção do resultado sem quebrar o fluxo
aba_selecao, aba_resultado = st.tabs(["👥 1. Mesa e Configuração", "🎬 2. Plenária / Resultado"])

# Controle ativo de abas via session_state
if st.session_state.aba_ativa == 0:
    with aba_selecao:
        # Filtros de busca lado a lado
        col_filtro_nome, col_filtro_partido = st.columns([2, 1])

        with col_filtro_nome:
            busca_nome = st.text_input("🔍 Filtrar por nome:", "", placeholder="Digite o nome do parlamentar...")

        with col_filtro_partido:
            lista_partidos = ["Todos"] + sorted(df_deputados['partido'].unique().tolist())
            partido_selecionado = st.selectbox("🎯 Filtrar por partido:", lista_partidos)

        # Filtros aplicados
        df_filtrado = df_deputados
        if busca_nome:
            df_filtrado = df_filtrado[df_filtrado['nome'].str.contains(busca_nome, case=False)]
        if partido_selecionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado['partido'] == partido_selecionado]

        # Renderização do Grid de Cards
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

else:
    with aba_resultado:
        if st.session_state.historico_debate is None:
            st.info("💤 Nenhuma simulação ativa na plenária. Monte a mesa na aba anterior e clique em 'Iniciar'.")
        else:
            dados_sessao = st.session_state.historico_debate
            
            st.subheader("📋 Pauta em Julgamento")
            st.caption(f"_{dados_sessao['pauta']}_")
            st.markdown(f"ℹ️ **Tags RAG Utilizadas:** `{dados_sessao['palavras_chave']}`")
            st.markdown("---")
            
            st.subheader("🎤 Pronunciamentos Oficiais")
            
            # Divide colunas dinâmicas dependendo de quantos deputados debateram
            qtd_deputados = len(dados_sessao['discursos'])
            cols_discursos = st.columns(qtd_deputados)
            
            for i, (id_dep, info) in enumerate(dados_sessao['discursos'].items()):
                with cols_discursos[i]:
                    st.chat_message("user", avatar=info['foto']).markdown(f"**{info['nome']} ({info['partido']})**")
                    st.markdown(f"> {info['discurso']}")
                    
            st.markdown("---")
            st.subheader("⚖️ Relatório de Mediação e Clima Político")
            st.info(dados_sessao['mediacao'])
            
            # Botão interno conveniente para retornar à aba de edição sem perder nada
            if st.button("⬅️ Voltar para Ajustar Mesa / Pauta"):
                st.session_state.aba_ativa = 0
                st.rerun()