import streamlit as st
import pandas as pd
import google.generativeai as genai

# Importações dos módulos locais
import config
import dados
import busca
import agentes

# Inicialização visual
config.configurar_pagina()

try:
    df_deputados = dados.carregar_df_deputados()
    vector_store, embeddings_model = busca.carregar_banco_vetorial()
except Exception as e:
    st.error(f"Erro ao inicializar arquivos ou modelos locais. Detalhes: {e}")
    st.stop()

# ==============================================================================
# MODAL DE AUDITORIA DE DISCURSOS
# ==============================================================================
@st.dialog("📚 Fontes Originais do RAG", width="large")
def exibir_modal_discursos(nome_deputado, lista_textos):
    st.markdown(f"Exibindo os discursos históricos reais recuperados para subsidiar a persona de **{nome_deputado}**.")
    st.markdown("---")
    if not lista_textos:
        st.info("Nenhum documento textual foi recuperado do banco vetorial para este parlamentar.")
    else:
        for idx, doc_text in enumerate(lista_textos, 1):
            with st.expander(f"📄 Fragmento Legislativo #{idx}", expanded=(idx == 1)):
                st.markdown(f"```text\n{doc_text}\n```")

# ==============================================================================
# ESTADOS DA SESSÃO (SESSION STATE)
# ==============================================================================
if 'selecionados' not in st.session_state: st.session_state.selecionados = []
if 'tela_atual' not in st.session_state: st.session_state.tela_atual = "config"
if 'debate_status' not in st.session_state: st.session_state.debate_status = "IDLE"
if 'ids_filtrados_tema' not in st.session_state: st.session_state.ids_filtrados_tema = None
if 'palavras_chave_persistentes' not in st.session_state: st.session_state.palavras_chave_persistentes = None
if 'historico_discursos_exibicao' not in st.session_state: st.session_state.historico_discursos_exibicao = {}
if 'cache_discursos' not in st.session_state: st.session_state.cache_discursos = {}
if 'cache_sentimentos' not in st.session_state: st.session_state.cache_sentimentos = {}
if 'cache_mediador' not in st.session_state: st.session_state.cache_mediador = ""
if 'cache_tags_orquestrador' not in st.session_state: st.session_state.cache_tags_orquestrador = ""
if 'modal_trigger_id' not in st.session_state: st.session_state.modal_trigger_id = None

# GATILHO DE MODAL
if st.session_state.modal_trigger_id is not None:
    dep_id = st.session_state.modal_trigger_id
    meta_m = df_deputados[df_deputados['id'] == dep_id].iloc[0]
    texts_m = st.session_state.historico_discursos_exibicao.get(dep_id, [])
    st.session_state.modal_trigger_id = None
    exibir_modal_discursos(meta_m['nome'], texts_m)

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

st.sidebar.markdown("### 🤖 Configuração dos Modelos IA")
modelo_orq_nome = st.sidebar.selectbox("Agente Analista de Relevância Temática:", config.MODELOS_DISPONIVEIS, index=0)
modelo_per_nome = st.sidebar.selectbox("Agentes Deputados:", config.MODELOS_DISPONIVEIS, index=2)
modelo_med_nome = st.sidebar.selectbox("Agente Mediador:", config.MODELOS_DISPONIVEIS, index=3)

modelo_orquestrador = agentes.inicializar_modelo(modelo_orq_nome)
modelo_persona      = agentes.inicializar_modelo(modelo_per_nome)
modelo_mediador     = agentes.inicializar_modelo(modelo_med_nome)

st.sidebar.markdown("---")
nova_pauta = st.sidebar.text_area(
    "📝 Proposta / Projeto de Lei:",
    value="Proposta de Emenda à Constituição para redução da maioridade penal para 16 anos em casos de crimes hediondos.",
    height=100
)

st.sidebar.markdown(f"### 👥 Mesa do Debate ({len(st.session_state.selecionados)} de {config.MAX_DEPUTADOS})")
if st.session_state.selecionados:
    for id_sel in st.session_state.selecionados:
        dep_info = df_deputados[df_deputados['id'] == id_sel].iloc[0]
        st.sidebar.caption(f"✅ {dep_info['nome']} ({dep_info['partido']})")
else:
    st.sidebar.info("Nenhum deputado selecionado.")

st.sidebar.markdown("---")
col_btn_iniciar, col_btn_reset = st.sidebar.columns(2)

with col_btn_iniciar:
    if st.button("🔥 Iniciar", type="primary", use_container_width=True, disabled=len(st.session_state.selecionados) == 0):
        st.session_state.tela_atual = "plenaria"
        st.session_state.debate_status = "RUNNING"
        st.session_state.historico_discursos_exibicao = {}
        st.session_state.cache_discursos = {}
        st.session_state.cache_sentimentos = {}
        st.session_state.cache_mediador = ""
        st.session_state.cache_tags_orquestrador = ""
        st.rerun()

with col_btn_reset:
    if st.button("♻️ Reiniciar", type="secondary", use_container_width=True):
        st.session_state.selecionados = []
        st.session_state.tela_atual = "config"
        st.session_state.debate_status = "IDLE"
        st.session_state.ids_filtrados_tema = None
        st.session_state.palavras_chave_persistentes = None
        st.session_state.historico_discursos_exibicao = {}
        st.session_state.cache_discursos = {}
        st.session_state.cache_sentimentos = {}
        st.session_state.cache_mediador = ""
        st.session_state.cache_tags_orquestrador = ""
        st.rerun()

# ==============================================================================
# CORPO PRINCIPAL E NAVEGAÇÃO DE TELAS
# ==============================================================================
st.title("🏛️ Arena de Debates Parlamentares Inteligente")

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
    col_filtro_nome, col_filtro_partido, col_filtro_tema = st.columns([2, 1, 1.5])

    with col_filtro_nome:
        busca_nome = st.text_input("🔍 Filtrar por nome:", "", placeholder="Digite o nome do parlamentar...")

    with col_filtro_partido:
        lista_partidos = ["Todos"] + sorted(df_deputados['partido'].unique().tolist())
        partido_selecionado = st.selectbox("🎯 Filtrar por partido:", lista_partidos)

    with col_filtro_tema:
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        col_btn_aplicar, col_btn_limpar = st.columns(2)
        
        with col_btn_aplicar:
            if st.button("📚 Filtrar por Tema", type="secondary", use_container_width=True):
                with st.spinner("🔍 Analisando histórico dos parlamentares..."):
                    termos_busca = agentes.agente_orquestrador(modelo_orquestrador, nova_pauta)
                    st.session_state.palavras_chave_persistentes = termos_busca
                    
                    ids_com_discurso = set()
                    for id_dep in df_deputados['id'].unique():
                        _, total_docs, _ = busca.buscar_contexto_deputado(id_dep, termos_busca, config.MIN_K_DOCUMENTOS)
                        if total_docs >= config.MIN_K_DOCUMENTOS:
                            ids_com_discurso.add(str(id_dep))
                    
                    st.session_state.ids_filtrados_tema = ids_com_discurso
                    st.rerun()
        
        with col_btn_limpar:
            if st.button("❌ Limpar Tema", type="secondary", use_container_width=True, disabled=st.session_state.ids_filtrados_tema is None):
                st.session_state.ids_filtrados_tema = None
                st.session_state.palavras_chave_persistentes = None
                st.rerun()

    df_filtrado = df_deputados
    if busca_nome:
        df_filtrado = df_filtrado[df_filtrado['nome'].str.contains(busca_nome, case=False)]
    if partido_selecionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['partido'] == partido_selecionado]
        
    if st.session_state.ids_filtrados_tema is not None:
        st.sidebar.warning("📚 Filtro por tema ativo!")
        df_filtrado = df_filtrado[df_filtrado['id'].isin(st.session_state.ids_filtrados_tema) | df_filtrado['id'].isin(st.session_state.selecionados)]

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
                    limite_atingido = len(st.session_state.selecionados) >= config.MAX_DEPUTADOS
                    if st.button("Selecionar", key=f"add_{id_dep}", type="primary", disabled=limite_atingido, use_container_width=True):
                        st.session_state.selecionados.append(id_dep)
                        st.rerun()

# --- TELA 2: PLENÁRIA ---
else:
    st.subheader("📋 Pauta em Julgamento")
    st.info(f"_{nova_pauta}_")
    
    if st.session_state.debate_status == "RUNNING":
        with st.spinner("🧠 Analista de Relevância Temática extraindo conceitos da pauta..."):
            if st.session_state.get('palavras_chave_persistentes'):
                palavras_chave = st.session_state.palavras_chave_persistentes
            else:
                palavras_chave = agentes.agente_orquestrador(modelo_orquestrador, nova_pauta)
            
        st.session_state.cache_tags_orquestrador = palavras_chave
        st.markdown(f"ℹ️ **Tags RAG Extraídas:** `{st.session_state.cache_tags_orquestrador}`")
        st.markdown("---")
        
        st.subheader("🎤 Pronunciamentos Oficiais e Postura Política")
        cols_discursos = st.columns(len(st.session_state.selecionados))

        for i, id_dep in enumerate(st.session_state.selecionados):
            meta = df_deputados[df_deputados['id'] == id_dep].iloc[0]
            chave_dep = f"{meta['nome']} ({meta['partido']})"
            
            with cols_discursos[i]:
                st.chat_message("user", avatar=meta['foto']).markdown(f"**{meta['nome']} ({meta['partido']})**")
                
                contexto_recuperado, total_docs, lista_textos_puros = busca.buscar_contexto_deputado(meta['id'], palavras_chave, config.K_DOCUMENTOS)
                st.session_state.historico_discursos_exibicao[meta['id']] = lista_textos_puros
                
                if total_docs == 0: st.warning("⚠️ Sem histórico localizado para este parlamentar. Atuando por diretriz partidária.")
                elif total_docs < config.MIN_K_DOCUMENTOS: st.caption(f"✨ Histórico real parcial integrado ({total_docs} discursos encontrados).")
                else: st.caption(f"✨ Histórico real completo integrado com sucesso ({total_docs} discursos encontrados).")
                    
                caixa_texto_dinamica = st.empty()
                caixa_sentimento = st.empty()
                
                repr_agente = agentes.AgenteRepresentante(meta['id'], meta['nome'], meta['partido'])
                resposta_stream = repr_agente.responder_a_pauta_stream(modelo_persona, nova_pauta, contexto_recuperado)
                
                texto_acumulado = ""
                for chunk in resposta_stream:
                    if chunk.text:
                        texto_acumulado += chunk.text
                        caixa_texto_dinamica.markdown(f"> {texto_acumulado}")
                
                with st.spinner("📊 Analisando tom político..."):
                    bg_color, text_color, rotulo_interno = dados.obter_cor_politica(texto_acumulado)
                
                caixa_sentimento.markdown(f"""
                    <div class="sentiment-box" style="background-color: {bg_color}; color: {text_color}; margin-top: 15px; border-left: 5px solid {text_color};">
                        {rotulo_interno.upper()}
                    </div>
                """, unsafe_allow_html=True)
                
                st.session_state.cache_discursos[chave_dep] = texto_acumulado
                st.session_state.cache_sentimentos[chave_dep] = rotulo_interno

        st.markdown("---")
        st.subheader("📊 Balanço Analítico do Plenário")
        
        contagem_tons = {"Altamente Opositor": 0, "Crítico": 0, "Neutro / Moderado": 0, "Favorável": 0, "Altamente Defensor": 0}
        for tom in st.session_state.cache_sentimentos.values():
            if tom in contagem_tons: contagem_tons[tom] += 1
                
        df_grafico = pd.DataFrame(list(contagem_tons.items()), columns=["Postura Política", "Quantidade de Deputados"])
        st.bar_chart(df_grafico, x="Postura Política", y="Quantidade de Deputados", color="#3b82f6")
        
        st.markdown("---")
        st.subheader("⚖️ Relatório de Mediação (Enriquecido com IA)")
        
        caixa_mediador_dinamica = st.empty()
        historico_mediador = ""
        for dep, txt in st.session_state.cache_discursos.items():
            tom_detectado = st.session_state.cache_sentimentos.get(dep, "Não detectado")
            historico_mediador += f"📌 [Pronunciamento de {dep} - Tom Político: {tom_detectado}]:\n{txt}\n\n"
            
        mediador_stream = agentes.agente_mediador(modelo_mediador, nova_pauta, historico_mediador)
        texto_med_acumulado = ""
        for chunk in mediador_stream:
            if chunk.text:
                texto_med_acumulado += chunk.text
                caixa_mediador_dinamica.info(texto_med_acumulado)
        
        st.session_state.cache_mediador = texto_med_acumulado
        st.session_state.debate_status = "FINISHED"
        st.rerun()

    elif st.session_state.debate_status == "FINISHED":
        st.markdown(f"ℹ️ **Tags RAG Extraídas:** `{st.session_state.cache_tags_orquestrador}`")
        st.markdown("---")
        st.subheader("🎤 Pronunciamentos Oficiais e Postura Política")
        cols_discursos = st.columns(len(st.session_state.selecionados))
        
        for i, id_dep in enumerate(st.session_state.selecionados):
            meta = df_deputados[df_deputados['id'] == id_dep].iloc[0]
            chave_dep = f"{meta['nome']} ({meta['partido']})"
            
            with cols_discursos[i]:
                st.chat_message("user", avatar=meta['foto']).markdown(f"**{meta['nome']} ({meta['partido']})**")
                
                total_docs = len(st.session_state.historico_discursos_exibicao.get(meta['id'], []))
                if total_docs == 0: st.warning("⚠️ Sem histórico localizado para este parlamentar. Atuando por diretriz partidária.")
                elif total_docs < config.MIN_K_DOCUMENTOS: st.caption(f"✨ Histórico real parcial integrado ({total_docs} discursos encontrados).")
                else: st.caption(f"✨ Histórico real completo integrado com sucesso ({total_docs} discursos encontrados).")
                
                discurso_salvo = st.session_state.cache_discursos.get(chave_dep, "")
                st.markdown(f"> {discurso_salvo}")
                
                tom_salvo = st.session_state.cache_sentimentos.get(chave_dep, "Não Analisado")
                bg_color, text_color, _ = dados.obter_cor_politica(discurso_salvo)
                st.markdown(f"""
                    <div class="sentiment-box" style="background-color: {bg_color}; color: {text_color}; margin-top: 15px; border-left: 5px solid {text_color};">
                        {tom_salvo.upper()}
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
                if st.button("📄 Inspecionar Fontes", key=f"src_stable_{meta['id']}", use_container_width=True):
                    st.session_state.modal_trigger_id = meta['id']
                    st.rerun()

        st.markdown("---")
        st.subheader("📊 Balanço Analítico do Plenário")
        contagem_tons = {"Altamente Opositor": 0, "Crítico": 0, "Neutro / Moderado": 0, "Favorável": 0, "Altamente Defensor": 0}
        for tom in st.session_state.cache_sentimentos.values():
            if tom in contagem_tons: contagem_tons[tom] += 1
        df_grafico = pd.DataFrame(list(contagem_tons.items()), columns=["Postura Política", "Quantidade de Deputados"])
        st.bar_chart(df_grafico, x="Postura Política", y="Quantidade de Deputados", color="#3b82f6")

        st.markdown("---")
        st.subheader("⚖️ Relatório de Mediação (Enriquecido com IA)")
        st.info(st.session_state.cache_mediador)
        st.success("💤 Sessão finalizada. Caso queira rodar uma nova simulação, retorne para a aba de Configuração ou altere a pauta.")