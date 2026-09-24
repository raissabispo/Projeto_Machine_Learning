# Aplicação de Detecção de Objetos utilizando YOLO e Visão Computacional

Aplicação de Visão Computacional para detecção automática de máscaras em imagens, vídeos e webcam utilizando YOLOv8n.

O projeto utiliza um modelo YOLOv8n treinado especificamente para identificar diferentes condições de uso de máscaras faciais.

---

## 1. Objetivo

Aplicar técnicas de Visão Computacional e Deep Learning para realizar a detecção automática de máscaras em diferentes tipos de entrada.

O projeto demonstra como um modelo YOLOv8n pré-treinado pode ser adaptado para uma aplicação específica por meio de treinamento com um dataset direcionado.

---

## 2. Detecção de Máscaras

O projeto utiliza um modelo YOLOv8n treinado para um problema específico de detecção de máscaras.

O modelo identifica três classes:

| Classe | Interpretação |
| ------------------------- | -------------------------------------- |
| `with_mask` | Pessoa com máscara |
| `without_mask` | Pessoa sem máscara |
| `mask_weared_incorrect` | Pessoa usando a máscara incorretamente |

A aplicação permite utilizar:

- imagens;
- vídeos;
- webcam.

Durante o uso da webcam, o sistema também fornece feedback por voz em português.

As frases utilizadas são:

- **"Pessoa com máscara."**
- **"Pessoa sem máscara."**
- **"Pessoa usando a máscara incorretamente."**

---

## 3. Cores das Detecções

Cada classe possui uma cor específica para facilitar a interpretação visual dos resultados:

| Classe | Cor |
| ------------------------- | -------- |
| `with_mask` | 🔵 Azul |
| `mask_weared_incorrect` | 🟡 Amarelo |
| `without_mask` | 🔴 Vermelho |

As cores são aplicadas às caixas delimitadoras e aos textos exibidos sobre os objetos detectados.

---

## 4. Contexto de Aplicação

A detecção de máscaras pode ser utilizada como uma ferramenta de apoio ao monitoramento do uso de proteção facial em ambientes onde esse tipo de equipamento é relevante.

Exemplos:

- hospitais e clínicas;
- laboratórios;
- ambientes industriais;
- construção;
- cozinhas profissionais;
- outros ambientes que utilizem máscaras como proteção.

---

## 5. Tecnologias Utilizadas

- Python
- YOLOv8n
- Ultralytics
- PyTorch
- OpenCV
- Streamlit
- Streamlit-WebRTC
- Edge TTS
- Pygame

---

## 6. Dataset

Para o projeto foi utilizado o dataset **Medical Mask Detection**, contendo três classes:

```text
mask_weared_incorrect
with_mask
without_mask
````

O conjunto utilizado no treinamento foi organizado em:

* **Treinamento:** 5.193 imagens
* **Validação:** 648 imagens
* **Teste:** 648 imagens

**Total:** 6.489 imagens.

---

## 7. Treinamento

O modelo utilizado para a detecção de máscaras foi o **YOLOv8n**.

Principais parâmetros utilizados:

| Parâmetro           |    Valor |
| ------------------- | -------: |
| Modelo              |  YOLOv8n |
| Épocas              |       30 |
| Tamanho das imagens |      640 |
| Batch               |       16 |
| GPU                 | Tesla T4 |

O treinamento foi realizado utilizando um conjunto de dados balanceado entre as três classes.

---


## 8. Arquitetura

```text
┌─────────────────┐
│     ENTRADA     │
│ Imagem / Vídeo  │
│ / Webcam        │
└────────┬────────┘
         ↓
┌─────────────────┐
│    BACKBONE     │
│ Extrai          │
│ características │
└────────┬────────┘
         ↓
┌─────────────────┐
│      NECK       │
│ Combina         │
│ características │
│ de diferentes   │
│ níveis          │
└────────┬────────┘
         ↓
┌─────────────────┐
│      HEAD       │
│ Faz a detecção  │
│ e classificação │
└────────┬────────┘
         ↓
┌─────────────────┐
│      SAÍDA      │
│ Classe + caixa  │
│ + confiança     │
└─────────────────┘
```

A mesma arquitetura YOLOv8n é utilizada para imagens, vídeos e webcam. O que muda é apenas a origem da entrada.

No modelo utilizado, o Backbone é formado pelas camadas iniciais de extração de características, o Neck realiza a combinação de características de diferentes níveis e o Head realiza a detecção final.

---

## 9. Estrutura do Projeto

```text
aplicacao-visao-computacional-yolo/

│
├── artifacts_yolo/
│
├── configuracao/
│   └── data.yaml
│
├── modelo/
│   └── best_mask_balanceado_final.pt
│
├── resultados/
│   ├── imagens/
│   └── resultados_video/
│
├── src/
│   ├── treinamento.py
│   ├── teste_imagem.py
│   ├── teste_video.py
│   ├── teste_tempo_real.py
│   └── teste_stremlit.py
│
├── testes/
│   ├── imagens_teste/
│   └── videos_teste/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 10. Principais Diretórios

### `modelo/`

Contém o modelo final treinado:

```text
best_mask_balanceado_final.pt
```

### `configuracao/`

Contém as configurações utilizadas pelo dataset:

```text
data.yaml
```

### `src/`

Contém os códigos principais de treinamento, testes e aplicação em tempo real.

Os principais arquivos são:

```text
treinamento.py
teste_imagem.py
teste_video.py
teste_tempo_real.py
teste_streamlit.py
```

### `testes/`

Contém as entradas utilizadas nas avaliações práticas:

```text
testes/

├── imagens_teste/
└── videos_teste/
```

### `resultados/`

Contém os resultados gerados pela aplicação:

```text
resultados/

├── imagens/
└── resultados_video/
```

### `artifacts_yolo/`

Contém artefatos relacionados ao treinamento e aos resultados do experimento.

---

## 11. Instalação

Clone o repositório:

```bash
git clone https://github.com/raissabispo/Projeto_YOLO_Machine_Learning.git
```

Entre na pasta:

```bash
cd Projeto_YOLO_Machine_Learning
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente no Windows:

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## 12. Execução

### Detecção em imagens

```bash
python src/teste_imagem.py
```

### Detecção em vídeos

```bash
python src/teste_video.py
```

### Detecção em tempo real com máscaras

```bash
python src/teste_tempo_real.py
```

### Aplicação Streamlit

A aplicação em tempo real utilizando webcam pode ser executada com:

```bash
streamlit run src/teste_stremlit.py
```

Após executar o comando, abra no navegador o endereço apresentado pelo Streamlit, normalmente:

```text
http://localhost:8501
```

---

## 13. Aplicação Streamlit

A aplicação Streamlit permite realizar a detecção de máscaras utilizando a webcam.

O fluxo da aplicação é:

```text
Webcam
   ↓
Streamlit-WebRTC
   ↓
Frame
   ↓
YOLOv8n
   ↓
Detecção
   ↓
Classe + Confiança
   ↓
Caixa colorida
   ↓
Exibição em tempo real
```

A interface apresenta:

* detecção pela webcam;
* caixas delimitadoras;
* classe detectada;
* confiança;
* quantidade de detecções;
* FPS;
* identificação por cores;
* feedback por voz.

A aplicação utiliza `streamlit-webrtc` para receber e processar os frames da webcam em tempo real.

---

## 14. Feedback por Voz

O sistema possui síntese de voz em português utilizando **Edge TTS** e reprodução do áudio com **Pygame**.

As mensagens são:

```text
Pessoa com máscara.

Pessoa sem máscara.

Pessoa usando a máscara incorretamente.
```

A aplicação utiliza uma etapa de estabilização das detecções para reduzir mudanças momentâneas de classe entre os frames.

O sistema considera os últimos 8 estados detectados e utiliza uma quantidade mínima de 6 ocorrências para definir um estado como estável.

A voz é acionada quando ocorre uma mudança de estado estável.

---

## 15. Estabilização das Detecções

Para reduzir oscilações entre diferentes classes durante a detecção em tempo real, o sistema mantém um histórico dos últimos 8 estados.

```text
Últimos 8 frames
       ↓
Contagem das classes
       ↓
Classe mais frequente
       ↓
Mínimo de 6 ocorrências
       ↓
Estado estável
```

Essa estratégia evita que uma pequena variação entre frames gere uma nova mensagem de voz a cada momento.

---

## 16. Limitações

Apesar dos resultados obtidos no conjunto de teste, o comportamento do modelo pode variar em situações diferentes das encontradas no dataset.

Entre as principais limitações estão:

* iluminação;
* distância da câmera;
* oclusões;
* posição do rosto;
* qualidade da imagem;
* diversidade dos dados;
* falsos positivos;
* falsos negativos.

Durante os testes em tempo real, também podem ocorrer falsos positivos em situações visualmente semelhantes às máscaras.

Os resultados obtidos no conjunto de teste não representam necessariamente o mesmo desempenho em todos os ambientes reais.

---

## 17. Considerações Finais

O projeto demonstra a utilização do YOLOv8n em uma aplicação específica de Visão Computacional: a detecção automática de máscaras.

O modelo identifica três condições diferentes:

1. pessoa com máscara;
2. pessoa sem máscara;
3. pessoa usando a máscara incorretamente.

A aplicação integra detecção em imagens, vídeos e webcam, além de fornecer feedback por voz em português durante a detecção em tempo real.

O projeto também apresenta uma interface desenvolvida com Streamlit para facilitar a utilização do modelo por meio do navegador.

O projeto foi desenvolvido como uma aplicação prática de técnicas de Machine Learning, Deep Learning e Visão Computacional.

## Licença e Dataset

Este projeto utiliza o dataset **Medical Mask Detection**,
disponibilizado sob a licença **CC BY 4.0**.

Dataset:
https://universe.roboflow.com/ai-workspace-yolo-ghcsl/medical-mask-detection/dataset/4

A utilização do dataset segue os termos da licença Creative Commons
Attribution 4.0 International (CC BY 4.0), com atribuição à fonte original.

Licença:
https://creativecommons.org/licenses/by/4.0/
