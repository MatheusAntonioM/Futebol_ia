#  Futebol IA: Motor de Análise Tática e Física

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=Ollama&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Um *pipeline* completo e automatizado de **Inteligência Artificial** para análise de partidas de futebol. Este sistema ingere vídeo bruto, utiliza Visão Computacional de ponta para extrair telemetria em tempo real (velocidade, distância, posse) e aplica raciocínio estratégico via LLM para gerar análises táticas automáticas e estruturadas.

> **Converte o movimento caótico de um jogo de futebol numa matriz matemática estruturada, pronta para análise profissional.**

---

##  📋 Índice
1. [Sobre o Projeto](#-sobre-o-projeto)
2. [Arquitetura e Pipeline](#-arquitetura-e-pipeline)
3. [Funcionalidades Principais](#-funcionalidades-principais)
4. [Fluxo de Dados](#-fluxo-de-dados)
5. [Tecnologias Utilizadas](#-tecnologias-utilizadas)
6. [Instalação e Configuração](#-instalação-e-configuração)
7. [Como Usar](#-como-usar)
8. [Estrutura do Projeto](#-estrutura-do-projeto)
9. [Troubleshooting](#-troubleshooting)

---

##  📖 Sobre o Projeto

Este projeto foi desenvolvido para demonstrar a integração seamless entre modelos de **Computer Vision** (para extração de dados físicos e espaciais) e **Generative AI** (para raciocínio estratégico sobre dados). 

**Objetivo:** Transformar um vídeo bruto de futebol numa análise tática estruturada, sem intervenção humana.

**Casos de Uso:**
- 📊 **Análise de Performance:** Identificar padrões de movimento e velocidades críticas
- 🎯 **Scouting Automático:** Avaliar jogadores com métricas quantificáveis
- 📈 **Estatísticas em Tempo Real:** Gerar telemetria contínua durante partidas
- 🤖 **Relatórios Automáticos:** Produzir análises tácticas estruturadas via IA

---

##  🏗️ Arquitetura e Pipeline

O fluxo de dados do sistema segue uma arquitetura linear de processamento em lote (*batch processing*):

```
┌─────────────────┐
│  Vídeo Bruto    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  1. INGESTÃO                                │
│  • Leitura frame a frame                    │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  2. DETEÇÃO E RASTREAMENTO                  │
│  • YOLO: Identifica jogadores, árbitros, bola │
│  • ByteTrack: Mantém IDs consistentes       │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  3. CALIBRAÇÃO ESPACIAL                     │
│  • Optical Flow: Anula movimento de câmara  │
│  • Perspectiva: Converte píxeis → metros   │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  4. EXTRAÇÃO LÓGICA                         │
│  • K-Means: Agrupa jogadores por cores      │
│  • Proximidade: Define posse de bola        │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  5. TELEMETRIA FÍSICA                       │
│  • Velocidade (km/h)                        │
│  • Distância percorrida (m)                 │
│  • Filtros de anomalias                     │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  6. RACIOCÍNIO LLM                          │
│  • Agregação JSON                           │
│  • LangChain → Gemma (Ollama)               │
│  • Análise tática estruturada                │
└────────┬────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  Relatório Automático │
│  (JSON + Análise)    │
└──────────────────────┘
```

### Descrição Detalhada das Etapas:

1. **Ingestão:** Leitura do vídeo frame a frame.

2. **Deteção e Rastreamento:** O modelo YOLO identifica os *bounding boxes* e o ByteTrack mantém o ID dos jogadores ao longo do tempo.

3. **Calibração Espacial:**
   - O *Optical Flow* anula o movimento de *pan* e *tilt* da câmara de transmissão.
   - Uma matriz de transformação de perspetiva converte as coordenadas 2D (píxeis) para um plano top-down (metros).

4. **Extração Lógica:** K-Means Clustering agrupa os jogadores por equipas com base nas cores dos equipamentos; a proximidade da bola define a posse.

5. **Telemetria Física:** Cálculo vetorial da velocidade e distância percorrida, com filtros de segurança contra anomalias visuais.

6. **Raciocínio LLM:** Agregação dos dados num dicionário JSON e envio via LangChain para o modelo Gemma (Ollama), que atua como analista tático.

---

##  ⚙️ Funcionalidades Principais

- **🎯 Deteção Multiclasse:** Rastreia Jogadores, Árbitros e a Bola simultaneamente (com interpolação de frames para momentos em que a bola é ocluída).

- **📹 Estabilização de Câmara:** Processamento de pontos de referência no relvado para isolar o movimento real do jogador do movimento da lente.

- **📐 Mapeamento 2D (Pixel to Meter):** Transformação geométrica que permite cálculos de física no mundo real, convertendo coordenadas de píxeis para metros.

- **⚡ Radar Físico e Limites Biológicos:** Velocidade calculada em km/h com suavização de dados e um travão de segurança arquitetural (`max = 40 km/h`) para ignorar falhas temporárias do YOLO.

- **📊 Relatório Automático (Pep Guardiola AI):** Utilização de *Prompt Engineering* sofisticado para redigir análises sobre "Controlo Estrutural", "Explosão Vertical" e "Eficiência Ofensiva" baseadas inteiramente na matemática dos dados capturados.

---

##  🔄 Fluxo de Dados

O JSON produzido pelo sistema tem a seguinte estrutura:

```json
{
  "frame": 120,
  "timestamp": "00:00:05.000",
  "players": [
    {
      "id": 1,
      "team": "A",
      "position": { "x": 52.5, "y": 34.2 },
      "velocity_kmh": 18.5,
      "distance_m": 245.3,
      "has_ball": false
    }
  ],
  "ball": {
    "position": { "x": 55.2, "y": 35.1 },
    "possession": "Team A - Player 7"
  },
  "tactical_insight": "Controlo estrutural: Team A mantém posse a 67%..."
}
```

---

##  🛠️ Tecnologias Utilizadas

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

##  📦 Instalação e Configuração

### Pré-requisitos
- **Python:** 3.8 ou superior
- **Sistema Operativo:** Mac, Linux ou Windows (recomenda-se GPU NVIDIA para processamento mais rápido do YOLO)
- **Ollama:** [Download aqui](https://ollama.com/) e instalar no seu sistema

### Passo 1: Clonar o Repositório

```bash
git clone https://github.com/MatheusAntonioM/Futebol_ia.git
cd Futebol_ia
```

### Passo 2: Criar Ambiente Virtual e Instalar Dependências

```bash
python -m venv venv

# No Windows:
venv\Scripts\activate

# No Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Passo 3: Configurar Ollama e o Modelo Gemma

```bash
# Fazer download do modelo Gemma
ollama pull gemma:2b

# Iniciar o servidor Ollama (em outro terminal)
ollama serve
```

O servidor Ollama funcionará por padrão em `http://localhost:11434`.

### Passo 4: Configurar Variáveis de Ambiente (opcional)

Crie um ficheiro `.env` na raiz do projeto:

```env
OLLAMA_BASE_URL=http://localhost:11434
YOLO_CONF=0.35
YOLO_IMGSZ=1088
MAX_SPEED_KMH=40
```

---

##  🚀 Como Usar

### Modo Básico: Processar um Vídeo

```bash
python main.py --video input_videos/seu_video.mp4 --output output_videos/
```

### Modo Avançado: Com Configurações Customizadas

```bash
python main.py \
  --video input_videos/partida.mp4 \
  --output output_videos/ \
  --conf 0.4 \
  --imgsz 1088 \
  --model models/best.pt \
  --skip-frames 2
```

### Parâmetros Disponíveis

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `--video` | str | **obrigatório** | Caminho para o vídeo de entrada |
| `--output` | str | `output_videos/` | Diretório de saída |
| `--conf` | float | `0.35` | Confiança YOLO (0-1) |
| `--imgsz` | int | `1088` | Tamanho das imagens YOLO |
| `--model` | str | `models/best.pt` | Caminho para o modelo YOLO |
| `--skip-frames` | int | `1` | Processar cada N frames |
| `--no-video` | bool | `False` | Apenas gerar JSON (sem vídeo) |

### Outputs Gerados

Após a execução, você terá em `output_videos/`:

```
output_videos/
├── seu_video_annotated.mp4       # Vídeo com anotações e tracking
├── seu_video_metrics.json        # Telemetria completa (frame-by-frame)
└── seu_video_tactical_report.txt # Análise tática gerada pela IA
```

### Exemplo de Uso Prático

```bash
# Processar um vídeo e gerar todos os outputs
python main.py --video input_videos/liverpool_vs_manchester.mp4

# Ver o resultado
# - Abrir: output_videos/liverpool_vs_manchester_annotated.mp4
# - Analisar: output_videos/liverpool_vs_manchester_tactical_report.txt
# - Dados brutos: output_videos/liverpool_vs_manchester_metrics.json
```

---

##  📂 Estrutura do Projeto

```
futebol_ia/
├── input_videos/                      # 📹 Ficheiros de vídeo brutos (.mp4)
├── output_videos/                     # 📊 Vídeos processados com anotações HUD
├── models/                            # 🤖 Modelos PyTorch do YOLO (best.pt)
├── stubs/                             # 💾 Ficheiros Pickle (.pkl) de cache
├── trackers/                          # 👁️ Deteção, tracking e renderização
│   └── tracker.py                     #    Orquestrador de deteção
├── main.py                            # 🎬 Orquestrador principal do Pipeline
├── tactical_analyst.py                # 📝 Módulo LangChain → Ollama
├── camera_movement_estimator.py       # 🎥 Lógica de estabilização de cena
├── view_transformer.py                # 🔄 Matriz de conversão Perspetiva/Metros
├── speed_and_distance_estimator.py    # ⚡ Motor de física (Velocidade e Distância)
├── team_assigner.py                   # 🎨 Machine Learning (K-Means) para cores
├── player_ball_assigner.py            # ⚽ Algoritmo de atribuição de posse
├── utils.py                           # 🛠️ Helpers (centros geométricos, medições)
├── requirements.txt                   # 📋 Dependências Python
├── .env.example                       # ⚙️ Template de configuração
└── README.md                          # 📖 Este ficheiro
```

---

##  🆘 Troubleshooting

### Problema: "CUDA out of memory"

**Causa:** A GPU não tem memória suficiente para processar imagens em alta resolução.

**Solução:**
```bash
# Reduza o tamanho das imagens YOLO
python main.py --video input.mp4 --imgsz 640

# Ou processe apenas alguns frames
python main.py --video input.mp4 --skip-frames 3
```

---

### Problema: "Ollama não consegue conectar"

**Causa:** O servidor Ollama não está a correr ou a URL está incorreta.

**Solução:**
```bash
# Verifique se o Ollama está a correr
ollama serve

# Verifique a URL em tactical_analyst.py
# Padrão: http://localhost:11434

# Se usar uma máquina remota, configure em .env:
# OLLAMA_BASE_URL=http://seu_servidor:11434
```

---

### Problema: "A bola não é detetada"

**Causa:** O modelo YOLO tem confiança muito alta ou o vídeo tem qualidade baixa.

**Solução:**
```bash
# Aumente a tolerância de confiança
python main.py --video input.mp4 --conf 0.25

# Ou use um modelo customizado treinado para bolas
python main.py --video input.mp4 --model models/custom_ball.pt
```

---

### Problema: "Vídeo de saída muito lento / Sistema trava"

**Causa:** Processamento muito pesado sem GPU ou vídeo muito longo.

**Solução:**
```bash
# Use GPU NVIDIA (se disponível)
# Verifique se CUDA está instalado

# Ou processe apenas a análise (sem renderizar vídeo)
python main.py --video input.mp4 --no-video

# Ou reduza a qualidade
python main.py --video input.mp4 --imgsz 480 --skip-frames 2
```

---

### Problema: "Jogadores trocam de cor / Team assigner errado"

**Causa:** O K-Means não conseguiu separar as cores corretamente (p.ex., uniformes muito similares).

**Solução:**
```python
# Edite team_assigner.py e ajuste o número de clusters
kmeans = KMeans(n_clusters=3)  # Tente 3 ou 4 em vez de 2

# Ou processe com interpolação de cores mais robusta
python main.py --video input.mp4 --conf 0.4
```

---

### Problema: "JSON vazio ou sem dados de telemetria"

**Causa:** Nenhum jogador foi detetado no vídeo.

**Solução:**
1. Verifique a qualidade do vídeo (resolução mínima: 720p)
2. Aumente a tolerância: `--conf 0.25`
3. Verifique se o modelo YOLO foi carregado corretamente
4. Experimente com um vídeo diferente para testar

---

## 📈 Resultados Esperados

Após processar um vídeo de 10 minutos (600 frames a 60 FPS):

| Métrica | Esperado |
|---------|----------|
| Tempo de processamento | ~2-5 minutos (com GPU) |
| Tamanho do JSON | ~50-100 MB |
| Frames detetados | 95%+ de sucesso |
| Velocidades calculadas | 0-40 km/h |

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Faça um Fork do repositório
2. Crie uma branch para a sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit as suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o ficheiro [LICENSE](LICENSE) para detalhes.

---

## 👨‍💻 Autor

**Matheus Antonio M**
- GitHub: [@MatheusAntonioM](https://github.com/MatheusAntonioM)
- 📧 Contacto: Disponível no perfil GitHub

---

## 🙏 Agradecimentos

- **Ultralytics** pela excelente implementação do YOLO
- **Google** pelo modelo Gemma
- **OpenAI** pela inspiração em análise de dados
- Comunidade open-source pelos frameworks utilizados

---

## 📚 Referências Úteis

- [YOLO Documentação](https://docs.ultralytics.com/)
- [LangChain Docs](https://python.langchain.com/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Supervision Library](https://github.com/roboflow/supervision)

---

**Última atualização:** Junho 2026
**Status:** ✅ Ativo e em desenvolvimento contínuo
