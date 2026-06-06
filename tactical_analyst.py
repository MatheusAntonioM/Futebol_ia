from langchain_community.llms.ollama import Ollama
import json

class TacticalAnalyst:
    def __init__(self, model_name="gemma4:latest"):
        # O LangChain trata de toda a configuração do URL do Ollama por trás dos panos!
        self.model_name = model_name
        self.llm = Ollama(model=self.model_name)

    def analyze(self, match_stats):
        print(f"\n A contactar o {self.model_name} via LangChain para análise tática...")
        
        prompt = f"""
        Aja como um analista tático de futebol de elite (como o Pep Guardiola).
        Acabei de processar um vídeo de um treino/jogo usando Visão Computacional.
        Com base nestas estatísticas finais rigorosas da partida, escreva um breve e empolgante 
        resumo tático da partida.
        
        Destaque:
        1. Qual foi a equipa dominante (baseado na posse de bola).
        2. A performance física impressionante (velocidade máxima e distância).

        Estatísticas Extraídas pela Inteligência Artificial:
        {json.dumps(match_stats, indent=2, ensure_ascii=False)}
        """

        try:
            # O LangChain envia o prompt e extrai apenas o texto final perfeitamente
            resposta = self.llm.invoke(prompt)
            return resposta
            
        except Exception as e:
            return f"Erro ao contactar a IA via LangChain. Verifique se a app do Ollama está aberta no computador. Detalhes: {e}"
