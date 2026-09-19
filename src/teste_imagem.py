from ultralytics import YOLO
import cv2
import os

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

pasta_imagens = os.path.join(
    pasta_projeto,
    "testes",
    "imagens_teste"
)


pasta_resultados = os.path.join(
    pasta_projeto,
    "resultados",
    "imagens"
)


os.makedirs(
    pasta_resultados,
    exist_ok=True
)



print("=" * 60)
print("CAMINHOS DO PROJETO")
print("=" * 60)

print("\nPasta do projeto:")
print(pasta_projeto)

print("\nModelo:")
print(caminho_modelo)

print("\nPasta de imagens:")
print(pasta_imagens)

print("\nPasta de resultados:")
print(pasta_resultados)



if not os.path.exists(caminho_modelo):

    print("\n❌ ERRO: modelo não encontrado!")
    print(caminho_modelo)

    exit()


if not os.path.exists(pasta_imagens):

    print("\n❌ ERRO: pasta de imagens não encontrada!")
    print(pasta_imagens)

    exit()



print("\nCarregando modelo...")

modelo = YOLO(
    caminho_modelo
)

print("✅ Modelo carregado!")
print()


arquivos = [
    arquivo
    for arquivo in os.listdir(pasta_imagens)
    if arquivo.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
]

arquivos.sort()


print(
    "Imagens encontradas:",
    len(arquivos)
)

print()



if len(arquivos) == 0:

    print(
        "❌ Nenhuma imagem encontrada na pasta:"
    )

    print(pasta_imagens)

    exit()



for numero, nome_arquivo in enumerate(
    arquivos,
    start=1
):

    print("=" * 60)

    print(
        f"Imagem {numero}/{len(arquivos)}: "
        f"{nome_arquivo}"
    )


    caminho_imagem = os.path.join(
        pasta_imagens,
        nome_arquivo
    )


    imagem = cv2.imread(
        caminho_imagem
    )


    if imagem is None:

        print(
            "❌ Erro ao abrir a imagem."
        )

        continue



    resultado = modelo(
        imagem,
        conf=0.25,
        verbose=False
    )[0]



    if len(resultado.boxes) == 0:

        print(
            "Nenhuma máscara detectada."
        )

    else:

        print(
            f"Detecções encontradas: "
            f"{len(resultado.boxes)}"
        )


        for caixa in resultado.boxes:

            # Classe
            classe = int(
                caixa.cls[0]
            )

            # Confiança
            confianca = float(
                caixa.conf[0]
            )

            # Nome da classe
            nome_classe = modelo.names[
                classe
            ]

            print(
                f"  {nome_classe} "
                f"- confiança: "
                f"{confianca:.2%}"
            )


    imagem_resultado = resultado.plot()


    caminho_resultado = os.path.join(
        pasta_resultados,
        nome_arquivo
    )


    cv2.imwrite(
        caminho_resultado,
        imagem_resultado
    )


    print(
        "✅ Resultado salvo em:"
    )

    print(
        caminho_resultado
    )


    cv2.imshow(
        "Teste do modelo - Mascaras",
        imagem_resultado
    )


    print(
        "\nPressione qualquer tecla "
        "para a próxima imagem."
    )

    print(
        "Pressione Q para encerrar."
    )


    tecla = (
        cv2.waitKey(0) & 0xFF
    )


    if tecla == ord("q"):

        break



cv2.destroyAllWindows()

print()

print("=" * 60)

print("✅ TESTE FINALIZADO!")

print("=" * 60)

print(
    "\nResultados salvos em:"
)

print(
    pasta_resultados
)