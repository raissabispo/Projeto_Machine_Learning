
# Aplicação de Detecção de Objetos utilizando YOLO e Visão Computacional

Aplicação de Visão Computacional para detecção automática de objetos em
imagens, vídeos e webcam utilizando modelos YOLO.

O projeto apresenta dois cenários:

- **Cenário 1:** detecção geral de objetos em tempo real utilizando um
  modelo YOLO pré-treinado.
- **Cenário 2:** detecção específica de máscaras utilizando um modelo
  YOLO treinado para identificar diferentes condições de uso da máscara.

---

## 1. Objetivo

Aplicar técnicas de Visão Computacional e Deep Learning para realizar
detecção automática de objetos em diferentes tipos de entrada.

O projeto também demonstra como um modelo YOLO pré-treinado pode ser
adaptado para uma aplicação específica por meio de treinamento com um
dataset direcionado.

---

## 2. Cenários do projeto

### Cenário 1 — Detecção geral de objetos

Utiliza um modelo YOLO pré-treinado para identificar diferentes objetos
em tempo real por meio da webcam.

A aplicação apresenta:

- caixas delimitadoras;
- nome do objeto;
- confiança da detecção;
- quantidade de objetos identificados;
- FPS.

O cenário está disponível em:

```text
webcam_deteccao_objetos/
````

---

### Cenário 2 — Detecção de máscaras

Utiliza um modelo YOLOv8n treinado para um problema específico de
detecção de máscaras.

O modelo identifica três classes:

| Classe                  | Interpretação                          |
| ----------------------- | -------------------------------------- |
| `with_mask`             | Pessoa com máscara                     |
| `without_mask`          | Pessoa sem máscara                     |
| `mask_weared_incorrect` | Pessoa usando a máscara incorretamente |

O cenário permite utilizar:

* imagens;
* vídeos;
* webcam.

Durante o uso da webcam, o sistema também fornece feedback por voz
em português.

As frases utilizadas são:

* **"Pessoa com máscara."**
* **"Pessoa sem máscara."**
* **"Pessoa usando a máscara incorretamente."**

---

## 3. Contexto de aplicação

A detecção de máscaras pode ser utilizada como ferramenta de apoio ao
monitoramento do uso de proteção facial em ambientes onde esse
equipamento é relevante.

Exemplos:

* hospitais e clínicas;
* laboratórios;
* ambientes industriais;
* construção;
* cozinhas profissionais;
* outros ambientes que utilizem máscaras como proteção.

---

## 4. Tecnologias utilizadas

* Python
* YOLOv8n
* Ultralytics
* PyTorch
* OpenCV
* Edge TTS
* Pygame

---

## 5. Dataset

Para o cenário de detecção de máscaras foi utilizado o dataset
**Medical Mask Detection**, contendo três classes:

```text
mask_weared_incorrect
with_mask
without_mask
```

O conjunto utilizado no treinamento foi organizado em:

* **Treinamento:** 5.193 imagens
* **Validação:** 648 imagens
* **Teste:** 648 imagens

**Total:** 6.489 imagens.

---

## 6. Treinamento

O modelo utilizado para a detecção de máscaras foi o **YOLOv8n**.

Principais parâmetros utilizados:

| Parâmetro           |    Valor |
| ------------------- | -------: |
| Modelo              |  YOLOv8n |
| Épocas              |       30 |
| Tamanho das imagens |      640 |
| Batch               |       16 |
| GPU                 | Tesla T4 |

O treinamento foi realizado utilizando um conjunto de dados balanceado
entre as três classes.

---

## 7. Resultados

### Resultado no conjunto de teste

| Métrica   | Resultado |
| --------- | --------: |
| Precisão  | **99,4%** |
| Recall    | **98,5%** |
| mAP@50    | **99,3%** |
| mAP@50-95 | **70,8%** |

### Resultado por classe

| Classe                  | Precisão | Recall | mAP@50 | mAP@50-95 |
| ----------------------- | -------: | -----: | -----: | --------: |
| `mask_weared_incorrect` |    99,0% |  97,2% |  99,4% |     67,6% |
| `with_mask`             |    99,3% |  98,6% |  98,9% |     74,8% |
| `without_mask`          |    99,9% |  99,5% |  99,5% |     70,0% |

Os resultados representam a avaliação realizada no conjunto de teste
utilizado no projeto.

---

## 8. Arquitetura

```text
                         APLICAÇÃO
                             |
              +--------------+--------------+
              |                             |
              v                             v
       CENÁRIO 1                      CENÁRIO 2
   Objetos gerais                      Máscaras
              |                             |
              v                             v
          Webcam                  Imagem / Vídeo / Webcam
              |                             |
              v                             v
          YOLOv8n                     YOLOv8n
       Pré-treinado                 Treinado para máscaras
              |                             |
              v                             v
     Objetos detectados             Máscaras detectadas
                                            |
                                            v
                                      Feedback por voz
```

---

## 9. Estrutura do projeto

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
│   └── teste_tempo_real.py
│
├── testes/
│   ├── imagens_teste/
│   └── videos_teste/
│
├── webcam_deteccao_objetos/
│
├── .gitignore
├── README.md
├── requirements.txt
├── yolo11n.pt
└── yolov8n.pt
```

---

## 10. Principais diretórios

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

Contém os códigos principais de treinamento e utilização do modelo.

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

Contém artefatos relacionados ao treinamento e aos resultados do
experimento.

### `webcam_deteccao_objetos/`

Contém a aplicação referente ao cenário de detecção geral de objetos.

---

## 11. Execução

### Instalação

Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
```

Entre na pasta:

```bash
cd aplicacao-visao-computacional-yolo
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

## 12. Executar os programas

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

### Detecção geral de objetos

A aplicação do cenário de detecção geral está disponível em:

```text
webcam_deteccao_objetos/
```

---

## 13. Feedback por voz

No cenário de detecção de máscaras, o sistema possui síntese de voz
em português.

As mensagens são:

```text
Pessoa com máscara.

Pessoa sem máscara.

Pessoa usando a máscara incorretamente.
```

A aplicação utiliza uma etapa de estabilização das detecções para
reduzir mudanças momentâneas de classe entre frames.

---

## 14. Limitações

Apesar dos resultados obtidos no conjunto de teste, o comportamento
do modelo pode variar em situações diferentes das encontradas no
dataset.

Entre as principais limitações estão:

* iluminação;
* distância da câmera;
* oclusões;
* posição do rosto;
* qualidade da imagem;
* diversidade dos dados;
* falsos positivos;
* falsos negativos.

Os resultados obtidos no conjunto de teste não representam
necessariamente o mesmo desempenho em todos os ambientes reais.

---

## 15. Considerações finais

O projeto demonstra a utilização de YOLO em dois cenários distintos:

1. detecção geral de objetos utilizando um modelo pré-treinado;
2. detecção específica de máscaras utilizando um modelo treinado para
   um problema direcionado.

A aplicação integra detecção por Visão Computacional com imagens,
vídeos e webcam, além de feedback por voz no cenário de máscaras.

O projeto foi desenvolvido como uma aplicação prática de técnicas de
Machine Learning, Deep Learning e Visão Computacional.

````
