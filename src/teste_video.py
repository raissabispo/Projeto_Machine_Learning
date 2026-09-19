from ultralytics import YOLO
import cv2
import os




pasta_src = os.path.dirname(
    os.path.abspath(__file__)
)

# Volta para a raiz do projeto:
# Projeto_Machine_Learning
pasta_projeto = os.path.dirname(
    pasta_src
)




caminho_modelo = os.path.join(
    pasta_projeto,
    "modelo",
    "best_mask_balanceado_final.pt"
)



pasta_videos = os.path.join(
    pasta_projeto,
    "testes",
    "videos_teste"
)



pasta_resultados = os.path.join(
    pasta_projeto,
    "resultados",
    "resultados_video"
)



os.makedirs(
    pasta_resultados,
    exist_ok=True
)


print("=" * 60)
print("CAMINHOS DO PROJETO")
print("=" * 60)

print("\nModelo:")
print(caminho_modelo)

print("\nPasta dos vídeos:")
print(pasta_videos)

print("\nPasta dos resultados:")
print(pasta_resultados)




if not os.path.exists(caminho_modelo):

    print("\n❌ Modelo não encontrado:")
    print(caminho_modelo)

    exit()




if not os.path.exists(pasta_videos):

    print("\n❌ Pasta de vídeos não encontrada:")
    print(pasta_videos)

    exit()


arquivos = [
    arquivo
    for arquivo in os.listdir(pasta_videos)
    if arquivo.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    )
]


# Ordena pelo nome
arquivos.sort()


print()
print(
    "Vídeos encontrados:",
    len(arquivos)
)



if len(arquivos) == 0:

    print(
        "\n❌ Nenhum vídeo encontrado em:"
    )

    print(
        pasta_videos
    )

    print(
        "\nArquivos encontrados nessa pasta:"
    )

    for arquivo in os.listdir(pasta_videos):

        print(
            " -",
            arquivo
        )

    exit()




print("\nCarregando modelo...")

modelo = YOLO(
    caminho_modelo
)

print("✅ Modelo carregado!")



for numero, nome_arquivo in enumerate(
    arquivos,
    start=1
):

    print()
    print("=" * 60)

    print(
        f"Vídeo {numero}/{len(arquivos)}: "
        f"{nome_arquivo}"
    )




    caminho_video = os.path.join(
        pasta_videos,
        nome_arquivo
    )



    video = cv2.VideoCapture(
        caminho_video
    )


    if not video.isOpened():

        print(
            "❌ Não foi possível abrir o vídeo."
        )

        continue



    largura = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    altura = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    quantidade_frames = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    # Caso o vídeo não informe FPS
    if fps <= 0:

        fps = 30


    print(
        f"Resolução: {largura}x{altura}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Frames: {quantidade_frames}"
    )



    nome_base = os.path.splitext(
        nome_arquivo
    )[0]

    nome_saida = (
        nome_base
        + "_resultado.mp4"
    )

    caminho_saida = os.path.join(
        pasta_resultados,
        nome_saida
    )




    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    saida = cv2.VideoWriter(
        caminho_saida,
        fourcc,
        fps,
        (largura, altura)
    )


    if not saida.isOpened():

        print(
            "❌ Não foi possível criar o vídeo de saída."
        )

        video.release()

        continue




    frame_atual = 0


    while True:


        sucesso, frame = video.read()


        if not sucesso:

            break


   

        resultado = modelo(
            frame,
            conf=0.40,
            verbose=False
        )[0]



        frame_resultado = resultado.plot()




        cv2.imshow(
            "YOLOv8 - Teste de Video",
            frame_resultado
        )


        saida.write(
            frame_resultado
        )


        frame_atual += 1


        if frame_atual % 100 == 0:

            print(
                f"Frames processados: "
                f"{frame_atual}/{quantidade_frames}"
            )


        # ----------------------------------------------------


        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            print(
                "\n⚠️ Vídeo interrompido pelo usuário."
            )

            break


    video.release()

    saida.release()

    cv2.destroyAllWindows()


    print()
    print(
        "✅ Vídeo processado!"
    )

    print(
        "Frames processados:",
        frame_atual
    )

    print(
        "Resultado salvo em:"
    )

    print(
        caminho_saida
    )


cv2.destroyAllWindows()


print()
print("=" * 60)

print(
    "✅ TESTE DOS VÍDEOS FINALIZADO!"
)

print("=" * 60)

print()
print(
    "Resultados salvos em:"
)

print(
    pasta_resultados
)