from ultralytics import YOLO

import edge_tts
import pygame

import cv2
import time
import asyncio
import threading
import uuid
import os

from collections import deque, Counter



pasta_src = os.path.dirname(
    os.path.abspath(__file__)
)


pasta_projeto = os.path.dirname(
    pasta_src
)




caminho_modelo = os.path.join(
    pasta_projeto,
    "modelo",
    "best_mask_balanceado_final.pt"
)




frases_mascaras = {

    "with_mask":
        "Pessoa com máscara.",

    "without_mask":
        "Pessoa sem máscara.",

    "mask_weared_incorrect":
        "Pessoa usando a máscara incorretamente."
}



print("=" * 60)
print("VERIFICAÇÃO DO PROJETO")
print("=" * 60)

print("\nPasta do projeto:")
print(pasta_projeto)

print("\nModelo:")
print(caminho_modelo)


if not os.path.exists(caminho_modelo):

    print("\n❌ ERRO: modelo não encontrado!")
    print(caminho_modelo)

    exit()


print("\nCarregando modelo...")

modelo = YOLO(
    caminho_modelo
)

print("✅ Modelo carregado!")

print(
    "Classes:",
    modelo.names
)

voz_falando = False


async def gerar_audio(texto, arquivo):

    comunicacao = edge_tts.Communicate(
        texto,
        "pt-BR-FranciscaNeural"
    )

    await comunicacao.save(arquivo)


def falar(texto):

    global voz_falando

    # Não inicia outra fala enquanto uma estiver tocando
    if voz_falando:
        return

    try:

        voz_falando = True

        # Nome único para o arquivo de áudio
        nome_audio = os.path.join(
            pasta_src,
            f"voz_{uuid.uuid4().hex}.mp3"
        )

        print(
            f"🔊 Falando: {texto}"
        )

        # Gera o áudio
        asyncio.run(
            gerar_audio(
                texto,
                nome_audio
            )
        )

        # Carrega o áudio
        pygame.mixer.music.load(
            nome_audio
        )

        # Reproduz
        pygame.mixer.music.play()

        # Aguarda terminar
        while pygame.mixer.music.get_busy():

            time.sleep(0.1)

        # Libera o áudio
        pygame.mixer.music.unload()

        # Remove arquivo temporário
        if os.path.exists(nome_audio):

            os.remove(nome_audio)

    except Exception as erro:

        print(
            f"❌ Erro na voz: {erro}"
        )

    finally:

        voz_falando = False



pygame.mixer.init()


camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print(
        "❌ Erro: não foi possível acessar a webcam."
    )

    pygame.quit()

    exit()


print("\n✅ Webcam iniciada!")
print("Pressione Q para sair.")



tempo_anterior = time.time()


# Guarda os últimos 8 estados
historico_estados = deque(
    maxlen=8
)

# A situação precisa aparecer pelo menos 6 vezes
# entre os últimos 8 frames
minimo_votos = 6


# Situação estável atual
estado_estavel = ()


# Última situação anunciada
ultimo_estado_falado = None


while True:


    sucesso, frame = camera.read()


    if not sucesso:

        print(
            "❌ Erro ao capturar imagem."
        )

        break


    resultados = modelo(
        frame,
        conf=0.40,
        verbose=False
    )

    resultado = resultados[0]


    frame_detectado = frame.copy()


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


        cv2.rectangle(
            frame_detectado,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )



        texto_caixa = (
            f"{frase} "
            f"{confianca:.2f}"
        )


        cv2.putText(
            frame_detectado,
            texto_caixa,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


    estado_atual = tuple(
        sorted(
            set(classes_detectadas)
        )
    )



    historico_estados.append(
        estado_atual
    )



    contador = Counter(
        historico_estados
    )


    estado_mais_frequente, quantidade = (
        contador.most_common(1)[0]
    )


    if (
        len(historico_estados) == 8
        and quantidade >= minimo_votos
    ):

        estado_estavel = (
            estado_mais_frequente
        )


    frases_estaveis = []


    for classe in estado_estavel:

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


    if texto_estavel:

        cv2.putText(
            frame_detectado,
            texto_estavel,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    if (
        estado_estavel
        and estado_estavel != ultimo_estado_falado
        and not voz_falando
    ):

        # Guarda o novo estado
        ultimo_estado_falado = (
            estado_estavel
        )


        # Cria a fala em uma thread
        threading.Thread(
            target=falar,
            args=(texto_estavel,),
            daemon=True
        ).start()



    tempo_atual = time.time()

    diferenca = (
        tempo_atual - tempo_anterior
    )


    if diferenca > 0:

        fps = 1 / diferenca

    else:

        fps = 0


    tempo_anterior = tempo_atual



    quantidade_objetos = len(
        resultado.boxes
    )


    cv2.putText(
        frame_detectado,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame_detectado,
        f"Deteccoes: {quantidade_objetos}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )



    cv2.imshow(
        "YOLOv8 - Deteccao de Mascaras",
        frame_detectado
    )



    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


camera.release()

cv2.destroyAllWindows()

pygame.quit()

print()
print("=" * 60)
print("✅ Programa encerrado.")
print("=" * 60)