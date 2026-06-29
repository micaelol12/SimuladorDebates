import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LC_FAISS

@st.cache_resource
def carregar_banco_vetorial():
    faiss_path = "./faiss_index"
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = LC_FAISS.load_local(faiss_path, embeddings_model, allow_dangerous_deserialization=True)
    return vector_store, embeddings_model

def buscar_contexto_deputado(id_deputado, termos_busca, k_docs):
    vector_store, embeddings_model = carregar_banco_vetorial()
    q_emb = embeddings_model.embed_query(termos_busca)
    
    docs_recuperados = vector_store.similarity_search_by_vector(
        embedding=q_emb, k=k_docs, filter={"id_deputado": str(id_deputado)}
    )
    lista_textos = [d.page_content for d in docs_recuperados]
    contexto_texto = "".join([f"<discurso>\n{txt}\n</discurso>\n\n" for txt in lista_textos])
    return contexto_texto, len(docs_recuperados), lista_textos