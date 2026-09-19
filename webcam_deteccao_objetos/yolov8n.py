from ultralytics import YOLO

from transformers import (
    MarianMTModel,
    MarianTokenizer
)

import edge_tts
import pygame

import cv2
import time
import asyncio
import threading
import uuid
import os




print("Carregando YOLOv8...")

modelo = YOLO("yolov8n.pt")

print("YOLOv8 carregado!")




print("Carregando modelo de tradução...")

nome_modelo_traducao = "Helsinki-NLP/opus-mt-tc-big-en-pt"

tokenizer = MarianTokenizer.from_pretrained(
    nome_modelo_traducao
)

modelo_tradutor = MarianMTModel.from_pretrained(
    nome_modelo_traducao
)

print("Modelo de tradução carregado!")




cache_traducao = {}


def traduzir(nome):
    

    if nome in cache_traducao:
        return cache_traducao[nome]

    try:

        tokens = tokenizer(
            [">>por<< " + nome],
            return_tensors="pt",
            padding=True
        )

        traducao = modelo_tradutor.generate(
            **tokens
        )

        resultado = tokenizer.decode(
            traducao[0],
            skip_special_tokens=True
        )

        # Guarda a tradução
        cache_traducao[nome] = resultado

        print(
            f"Tradução automática: "
            f"{nome} → {resultado}"
        )

        return resultado

    except Exception as erro:

        print(
            f"Erro na tradução de {nome}: {erro}"
        )

        return nome



voz_falando = False


async def gerar_audio(texto, arquivo):

    comunicacao = edge_tts.Communicate(
        texto,
        "pt-BR-FranciscaNeural"
    )

    await comunicacao.save(arquivo)


def falar(texto):

    global voz_falando

    if voz_falando:
        return

    try:

        voz_falando = True

        # Nome único para o áudio
        nome_audio = (
            f"voz_{uuid.uuid4().hex}.mp3"
        )

        print(f"🔊 Falando: {texto}")

        # Gera o áudio
        asyncio.run(
            gerar_audio(
                texto,
                nome_audio
            )
        )

        # Reproduz
        pygame.mixer.music.load(
            nome_audio
        )

        pygame.mixer.music.play()

        # Espera terminar
        while pygame.mixer.music.get_busy():

            time.sleep(0.1)

        pygame.mixer.music.unload()

        # Apaga arquivo temporário
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

    exit()

print("Webcam iniciada!")
print("Pressione Q para sair.")


tempo_anterior = time.time()



ultimo_texto = ""

ultimo_anuncio = 0

intervalo_fala = 3



while True:

    sucesso, frame = camera.read()

    if not sucesso:

        print(
            "❌ Erro ao capturar imagem."
        )

        break


    resultados = modelo(
        frame,
        conf=0.25,
        verbose=False
    )

    resultado = resultados[0]




    frame_detectado = frame.copy()



    objetos = []



    for caixa in resultado.boxes:

        # Classe
        classe = int(
            caixa.cls[0]
        )

        # Confiança
        confianca = float(
            caixa.conf[0]
        )

        # Nome em inglês
        nome_ingles = modelo.names[
            classe
        ]


        nome_portugues = traduzir(
            nome_ingles
        )


        # Guardar objeto
        objetos.append(
            nome_portugues
        )


        x1, y1, x2, y2 = map(
            int,
            caixa.xyxy[0]
        )


    

        cv2.rectangle(
            frame_detectado,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )



        texto_caixa = (
            f"{nome_portugues} "
            f"{confianca:.2f}"
        )


        cv2.putText(
            frame_detectado,
            texto_caixa,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )




    objetos_unicos = list(
        dict.fromkeys(objetos)
    )



    if objetos_unicos:

        texto = (
            "Objetos detectados: "
            + ", ".join(objetos_unicos)
        )

    else:

        texto = ""



    if texto:

        cv2.putText(
            frame_detectado,
            texto,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )



    tempo_atual = time.time()

    fps = 1 / (
        tempo_atual - tempo_anterior
    )

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
        f"Objetos: {quantidade_objetos}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )



    tempo_passado = (
        time.time() - ultimo_anuncio
    )


    mudou = (
        texto != ultimo_texto
    )


    passou_tempo = (
        tempo_passado >= intervalo_fala
    )


    if (
        texto
        and (mudou or passou_tempo)
        and not voz_falando
    ):

        ultimo_texto = texto

        ultimo_anuncio = time.time()


        threading.Thread(
            target=falar,
            args=(texto,),
            daemon=True
        ).start()


    # ========================================================
    # MOSTRAR WEBCAM
    # ========================================================

    cv2.imshow(
        "YOLOv8 - Deteccao em tempo real",
        frame_detectado
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):

        break



camera.release()

cv2.destroyAllWindows()

pygame.quit()

print("Programa encerrado.")