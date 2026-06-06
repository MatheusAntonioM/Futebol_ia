#  Futebol IA: Motor de Análise Tática e Física

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=Ollama&logoColor=white)

Um *pipeline* completo e automatizado de **Inteligência Artificial** para análise de partidas de futebol. Este sistema ingere vídeo bruto, utiliza Visão Computacional de ponta para extrair telemetria precisa em tempo real e alimenta um modelo de linguagem local (LLM) para gerar um relatório tático profundo e narrativo.

---

##  Índice
1. [Sobre o Projeto](#-sobre-o-projeto)
2. [Arquitetura e Pipeline](#-arquitetura-e-pipeline)
3. [Funcionalidades Principais](#-funcionalidades-principais)
4. [Tecnologias Utilizadas](#-tecnologias-utilizadas)
5. [Instalação e Configuração](#-instalação-e-configuração)
6. [Como Usar](#-como-usar)
7. [Estrutura do Projeto](#-estrutura-do-projeto)
8. [Troubleshooting (Resolução de Problemas)](#-troubleshooting)

---

##  Sobre o Projeto
Este projeto foi desenvolvido para demonstrar a integração entre modelos de **Computer Vision** (para extração de dados físicos e espaciais) e **Generative AI** (para raciocínio sobre dados). Ele converte o movimento caótico de um jogo de futebol numa matriz matemática estruturada e, em seguida, traduz essa matemática num relatório tático compreensível por humanos.

##  Arquitetura e Pipeline
O fluxo de dados do sistema segue uma arquitetura linear de processamento em lote (*batch processing*):

1. **Ingestão:** Leitura do vídeo frame a frame.
2. **Deteção e Rastreamento:** O modelo YOLO identifica os *bounding boxes* e o ByteTrack mantém o ID dos jogadores ao longo do tempo.
3. **Calibração Espacial:**
   - O *Optical Flow* anula o movimento de *pan* e *tilt* da câmara de transmissão.
   - Uma matriz de transformação de perspetiva converte as coordenadas 2D (píxeis) para um plano top-down (metros).
4. **Extração Lógica:** K-Means Clustering agrupa os jogadores por equipas com base nas cores dos equipamentos; a proximidade da bola define a posse.
5. **Telemetria Física:** Cálculo vetorial da velocidade e distância percorrida, com filtros de segurança contra anomalias visuais.
6. **Raciocínio LLM:** Agregação dos dados num dicionário JSON e envio via LangChain para o modelo Gemma (Ollama), que atua como analista tático.

---

##  Funcionalidades Principais
- **Deteção Multiclasse:** Rastreia Jogadores, Árbitros e a Bola simultaneamente (com interpolação de frames para momentos em que a bola é ocluída).
- **Estabilização de Câmara:** Processamento de pontos de referência no relvado para isolar o movimento real do jogador do movimento da lente.
- **Mapeamento 2D (Pixel to Meter):** Transformação geométrica que permite cálculos de física no mundo real.
- **Radar Físico e Limites Biológicos:** Velocidade calculada em km/h com suavização e um travão de segurança arquitetural (`max = 40 km/h`) para ignorar falhas temporárias do YOLO.
- **Relatório Automático (Pep Guardiola AI):** Utilização de *Prompt Engineering* para redigir análises sobre "Controlo Estrutural" e "Explosão Vertical" baseadas inteiramente na matemática da partida.

---

##  Tecnologias Utilizadas

### Visão Computacional & Física
* **YOLO (Ultralytics):** Deteção de objetos (`conf=0.35`, `imgsz=1088` para alta precisão).
* **Supervision:** *Tracking* (ByteTrack) e anotações visuais dinâmicas no vídeo.
* **OpenCV (`cv2`):** Processamento de imagem, *Optical Flow* e desenho de geometrias.
* **NumPy & Pandas:** Cálculos vetoriais, médias móveis e interpolação de dados nulos.

### Inteligência Artificial (LLM)
* **Ollama:** Servidor local de inferência para modelos Open-Source.
* **Gemma 4:** Modelo fundacional da Google, utilizado como motor lógico.
* **LangChain:** Framework de orquestração para construir a ponte entre os dados em Python e o prompt do LLM.

---

##  Instalação e Configuração

### Pré-requisitos
- Python 3.8 ou superior.
- Mac, Linux ou Windows (recomenda-se GPU para processamento mais rápido do YOLO).
- [Ollama](https://ollama.com/) instalado no sistema operativo.


## Estrutura do Projeto   

futebol_ia/
* ├── input_videos/                  # Ficheiros de vídeo brutos (.mp4)
* ├── output_videos/                 # Vídeos processados com anotações HUD
* ├── models/                        # Modelos PyTorch do YOLO (best.pt)
* ├── stubs/                         # Ficheiros Pickle (.pkl) de cache
* ├── trackers/                      # Deteção, tracking e renderização
* │   └── tracker.py
* ├── main.py                        # Orquestrador do Pipeline
* ├── tactical_analyst.py            # Módulo LangChain -> Ollama
* ├── camera_movement_estimator.py   # Lógica de estabilização de cena
* ├── view_transformer.py            # Matriz de conversão Perspetiva/Metros
* ├── speed_and_distance_estimator.py # Motor de física (Velocidade e Distância)
* ├── team_assigner.py               # Machine Learning (K-Means) para cores
* ├── player_ball_assigner.py        # Algoritmo de atribuição de posse
* └── utils.py                       # Helpers (centros geométricos, medições)
