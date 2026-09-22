import os
import time
import asyncio
import threading
import uuid
from collections import deque, Counter

import av
import cv2
import edge_tts
import pygame
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer




st.set_page_config(
    page_title="YOLOv8 - Detecção de Máscaras",
    page_icon="😷",
    layout="wide"
)




st.markdown(
    """
    <style>

    /* =====================================================
       FUNDO PRINCIPAL
       ===================================================== */

    .stApp {
        background-color: #EAF3FF;
        color: #0F2D52;
    }


    /* =====================================================
       BARRA SUPERIOR
       ===================================================== */

    [data-testid="stHeader"] {
        background-color: #EAF3FF;
    }


    /* =====================================================
       TÍTULO
       ===================================================== */

    h1 {
        color: #0B4F9C;
        font-weight: 700;
    }


    /* =====================================================
       SUBTÍTULOS
       ===================================================== */

    h2,
    h3 {
        color: #123F73;
    }


    /* =====================================================
       TEXTOS
       ===================================================== */

    p,
    label,
    .stMarkdown,
    [data-testid="stMarkdownContainer"] {
        color: #123F73;
    }


    /* =====================================================
       CAIXAS DE INFORMAÇÃO
       ===================================================== */

    .stAlert {
        background-color: #D6E9FF;
        border-radius: 10px;
        color: #123F73;
    }


    /* =====================================================
       MÉTRICAS
       ===================================================== */

    [data-testid="stMetric"] {
        background-color: #D6E9FF;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
    }


    [data-testid="stMetricLabel"] {
        color: #1E4E85;
    }


    [data-testid="stMetricValue"] {
        color: #0B4F9C;
    }


    /* =====================================================
       BOTÕES
       ===================================================== */

    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }


    .stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }


    /* =====================================================
       ÁREA DA WEBCAM
       ===================================================== */

    [data-testid="stCameraInput"] {
        background-color: #D6E9FF;
    }


    /* =====================================================
       RODAPÉ
       ===================================================== */

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TÍTULO
# ============================================================

st.title(
    "Detecção de Máscaras em Tempo Real"
)

st.write(
    "Aplicação de Visão Computacional utilizando YOLOv8n "
    "para detecção de máscaras pela webcam."
)




pasta_src = os.path.dirname(
    os.path.abspath(__file__)
)

# Volta de src/ para a raiz do projeto
pasta_projeto = os.path.dirname(
    pasta_src
)

# Caminho do modelo
caminho_modelo = os.path.join(
    pasta_projeto,
    "modelo",
    "best_mask_balanceado_final.pt"
)



if not os.path.exists(caminho_modelo):

    st.error(
        "❌ Modelo não encontrado."
    )

    st.write(
        "Caminho procurado:"
    )

    st.code(
        caminho_modelo
    )

    st.stop()



@st.cache_resource
def carregar_modelo():

    modelo = YOLO(
        caminho_modelo
    )

    return modelo


modelo = carregar_modelo()


frases_mascaras = {

    "with_mask":
        "Pessoa com máscara.",

    "without_mask":
        "Pessoa sem máscara.",

    "mask_weared_incorrect":
        "Pessoa usando a máscara incorretamente."

}




cores_mascaras = {

    # Azul
    "with_mask":
        (255, 120, 0),

    # Amarelo
    "mask_weared_incorrect":
        (0, 255, 255),

    # Vermelho
    "without_mask":
        (0, 0, 255)

}




if not pygame.mixer.get_init():

    pygame.mixer.init()


# Impede duas falas simultâneas
voz_lock = threading.Lock()



async def gerar_audio(
    texto,
    arquivo
):

    comunicacao = edge_tts.Communicate(
        texto,
        "pt-BR-FranciscaNeural"
    )

    await comunicacao.save(
        arquivo
    )



def falar(
    texto
):


    if not voz_lock.acquire(
        blocking=False
    ):
        return



    nome_audio = os.path.join(
        pasta_src,
        f"voz_{uuid.uuid4().hex}.mp3"
    )


    try:

    
        asyncio.run(
            gerar_audio(
                texto,
                nome_audio
            )
        )




        pygame.mixer.music.load(
            nome_audio
        )


        pygame.mixer.music.play()



        while pygame.mixer.music.get_busy():

            time.sleep(
                0.1
            )



        pygame.mixer.music.stop()



        try:

            pygame.mixer.music.unload()

        except Exception:

            pass


    except Exception as erro:

        print(
            f"❌ Erro na voz: {erro}"
        )


    finally:

        if os.path.exists(
            nome_audio
        ):

            try:

                os.remove(
                    nome_audio
                )

            except Exception:

                pass



        voz_lock.release()


class ProcessadorVideo:

    def __init__(
        self
    ):

        self.historico_estados = deque(
            maxlen=8
        )


        self.estado_estavel = ()

        self.minimo_votos = 6


        self.ultimo_estado_falado = None



        self.tempo_anterior = time.time()



    def processar_frame(
        self,
        frame
    ):

        

        imagem = frame.to_ndarray(
            format="bgr24"
        )




        resultados = modelo(
            imagem,
            conf=0.40,
            verbose=False
        )

        resultado = resultados[0]


    

        frame_detectado = imagem.copy()




        classes_detectadas = []


   

        for caixa in resultado.boxes:



            classe = int(
                caixa.cls[0]
            )


     

            confianca = float(
                caixa.conf[0]
            )


         

            nome_classe = modelo.names[
                classe
            ]




            classes_detectadas.append(
                nome_classe
            )


  

            x1, y1, x2, y2 = map(
                int,
                caixa.xyxy[0]
            )


           

            frase = frases_mascaras.get(
                nome_classe,
                "Condição da máscara não identificada."
            )


      
            cor = cores_mascaras.get(
                nome_classe,
                (255, 255, 255)
            )


    

            cv2.rectangle(
                frame_detectado,
                (x1, y1),
                (x2, y2),
                cor,
                2
            )


   

            texto_caixa = (
                f"{frase} "
                f"{confianca:.2f}"
            )


            cv2.putText(
                frame_detectado,
                texto_caixa,
                (
                    x1,
                    max(
                        y1 - 10,
                        25
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                cor,
                2
            )


    

        estado_atual = tuple(
            sorted(
                set(
                    classes_detectadas
                )
            )
        )


  

        self.historico_estados.append(
            estado_atual
        )


     

        contador = Counter(
            self.historico_estados
        )


        if contador:

            estado_mais_frequente, quantidade = (
                contador.most_common(1)[0]
            )


         

            if (
                len(
                    self.historico_estados
                ) == 8

                and

                quantidade >= self.minimo_votos
            ):

                self.estado_estavel = (
                    estado_mais_frequente
                )



        posicao_y = 120

        for classe in self.estado_estavel:

            frase = frases_mascaras.get(
                classe,
                "Condição da máscara não identificada."
            )

            cor = cores_mascaras.get(
                classe,
                (255, 255, 255)
            )

            cv2.putText(
                frame_detectado,
                frase,
                (
                    20,
                    posicao_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                cor,
                2
            )

            posicao_y += 35


   

        if (
            self.estado_estavel

            and

            self.estado_estavel
            != self.ultimo_estado_falado
        ):

        
            self.ultimo_estado_falado = (
                self.estado_estavel
            )



            frases_estaveis = []

            for classe in self.estado_estavel:

                frase = frases_mascaras.get(
                    classe,
                    "Condição da máscara não identificada."
                )

                frases_estaveis.append(
                    frase
                )


            texto_estavel = " ".join(
                frases_estaveis
            )


          

            threading.Thread(
                target=falar,
                args=(texto_estavel,),
                daemon=True
            ).start()


   

        tempo_atual = time.time()

        diferenca = (
            tempo_atual
            -
            self.tempo_anterior
        )


        if diferenca > 0:

            fps = 1 / diferenca

        else:

            fps = 0


        self.tempo_anterior = (
            tempo_atual
        )


  

        quantidade_objetos = len(
            resultado.boxes
        )


       

        cv2.putText(
            frame_detectado,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )


    

        cv2.putText(
            frame_detectado,
            f"Deteccoes: {quantidade_objetos}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )



        return av.VideoFrame.from_ndarray(
            frame_detectado,
            format="bgr24"
        )





processador = ProcessadorVideo()




st.subheader(
    "Detecção em tempo real"
)

st.info(
    "Clique em START para iniciar a câmera. "
    "O sistema também fornece feedback por voz."
)




webrtc_streamer(
    key="deteccao-mascaras",

    video_frame_callback=(
        processador.processar_frame
    ),

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True
)



st.markdown("---")

st.subheader(
    "Legenda das detecções"
)

st.markdown(
    """
    🔵 **Azul:** Pessoa com máscara  
    🟡 **Amarelo:** Pessoa usando a máscara incorretamente  
    🔴 **Vermelho:** Pessoa sem máscara
    """
)





st.markdown("---")

st.subheader(
    "Informações do modelo"
)


coluna1, coluna2, coluna3 = st.columns(
    3
)


with coluna1:

    st.metric(
        "Modelo",
        "YOLOv8n"
    )


with coluna2:

    st.metric(
        "Confiança mínima",
        "0.40"
    )


with coluna3:

    st.metric(
        "Classes",
        "3"
    )


st.caption(
    "with_mask • mask_weared_incorrect • without_mask"
)