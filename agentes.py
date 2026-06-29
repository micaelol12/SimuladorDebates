import google.generativeai as genai
from busca import buscar_contexto_deputado

def inicializar_modelo(nome_modelo):
    return genai.GenerativeModel(nome_modelo)

def agente_orquestrador(modelo, pauta: str):
    prompt = f"Você é o Orquestrador Técnico da Câmara. Extraia termos de busca e palavras-chave separados por vírgula ideais para buscar discursos antigos pertinentes a esta pauta: {pauta}. Não faça perguntas e não apresente dicas."
    return modelo.generate_content(prompt).text.strip()

def agente_mediador(modelo, pauta, historico_mediador):
    prompt_med = f"Você é o Mediador da Câmara. Avalie o debate gerado sobre a pauta '{pauta}'. Leve em consideração o tom político detectado em cada discurso. Elabore um relatório contendo: 1) Pontos de convergência, 2) Conflitos ideológicos e 3) Clima Político geral.\n\nDebate:\n{historico_mediador}"
    return modelo.generate_content(prompt_med, stream=True)

class AgenteRepresentante:
    def __init__(self, id_deputado, nome, partido):
        self.id_deputado = str(id_deputado)
        self.nome = nome
        self.partido = partido

    def responder_a_pauta_stream(self, modelo, pauta, contexto):
        prompt_persona = f"""
        Você é o parlamentar {self.nome} do partido {self.partido}. 
        Simule seu pronunciamento oficial em 1ª pessoa na Tribuna da Câmara sobre a seguinte pauta: {pauta}.
        Baseie-se fortemente no seu histórico de discursos fornecido abaixo. Caso o histórico seja escasso ou não mencione diretamente o assunto, adote uma postura ideológica coerente com seu partido ({self.partido}).
        
        Histórico Real Recuperado: 
        {contexto if contexto else "Nenhum histórico específico encontrado."}
        """
        return modelo.generate_content(prompt_persona, stream=True)