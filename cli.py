# cli.py
import pandas as pd
from pre_procesamiento import preprocesar
from indice_invertido import construir_indice
from modelo_ri import IRModelIndex
from tqdm import tqdm

CSV_PATH = "data/output.csv"

def cargar_y_preparar(csv_path=CSV_PATH):
    print("Cargando corpus...")
    df = pd.read_csv(csv_path)


    # LIMITAR CORPUS A 20 000 DOCUMENTOS
    df = df.head(20000)

    df.rename(columns={'content': 'review_text'}, inplace=True)
    df.set_index('id', inplace=True)
    df.index.name = "doc_id"

    df["review_text"] = df["review_text"].fillna("").astype(str)

    print("\nPreprocesando documentos (esto puede tardar un momento)...")

    tokens_list = []
    for texto in tqdm(df["review_text"], desc="Progreso", unit="doc"):
        tokens_list.append(preprocesar(texto))

    df["tokens"] = tokens_list

    print("Preprocesamiento completado.\n")
    return df

def preview_text(texto, length=200):
    texto = texto.replace("\n", " ").strip()
    return texto[:length] + ("..." if len(texto) > length else "")

def main():
    df = cargar_y_preparar()
    print(f"Corpus cargado. Documentos: {len(df)}")

    print("Construyendo índice invertido...")
    indice = construir_indice(df)
    print("Índice invertido construido.")

    model = IRModelIndex(indice, df.index.tolist())

    print("   \n SISTEMA DE RI - REVIEWS STEAM  ")


    while True:
        query = input("\nConsulta > ").strip()

        if query.lower() == "salir":
            print("Adiós")
            break

        tokens_q = preprocesar(query)

        # JACCARD 
        print("\n TOP 10 JACCARD ")
        res_j = model.consulta_jaccard(tokens_q)
        for r, (doc, score) in enumerate(res_j, start=1):
            texto_preview = preview_text(df.loc[doc, "review_text"])
            print(f"\n{r}. Documento {doc} | Score={score:.4f}")
            print(f"   Texto: {texto_preview}")

        # TF-IDF 
        print("\n TOP 10 TF-IDF ")
        res_t = model.consulta_tfidf(tokens_q)
        for r, (doc, score) in enumerate(res_t, start=1):
            texto_preview = preview_text(df.loc[doc, "review_text"])
            print(f"\n{r}. Documento {doc} | Score={score:.4f}")
            print(f"   Texto: {texto_preview}")

        #  BM25
        print("\n TOP 10 BM25")
        res_b = model.consulta_bm25(tokens_q)
        for r, (doc, score) in enumerate(res_b, start=1):
            texto_preview = preview_text(df.loc[doc, "review_text"])
            print(f"\n{r}. Documento {doc} | Score={score:.4f}")
            print(f"   Texto: {texto_preview}")

if __name__ == "__main__":
    main()
