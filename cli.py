# cli.py
import pandas as pd
import json
from pre_procesamiento import preprocesar, construir_indice
from modelo_ri import IRModelIndex
from tqdm import tqdm

CSV_PATH = "data/output.csv"
QRELS_PATH = "qrels.json"

def cargar_y_preparar(csv_path=CSV_PATH):
    print("Cargando corpus...")
    df = pd.read_csv(csv_path)

    df = df.head(20000)

    df.rename(columns={'content': 'review_text'}, inplace=True)
    df.set_index('id', inplace=True)
    df.index.name = "doc_id"

    df["review_text"] = df["review_text"].fillna("").astype(str)

    print("\nPreprocesando documentos...")
    tokens_list = []
    for texto in tqdm(df["review_text"], desc="Progreso", unit="doc"):
        tokens_list.append(preprocesar(texto))

    df["tokens"] = tokens_list
    print("Preprocesamiento completado.\n")
    return df


def preview_text(texto, length=200):
    texto = texto.replace("\n", " ").strip()
    return texto[:length] + ("..." if len(texto) > length else "")



def precision_at_k(resultados, relevantes, k=10):
    retrieved = [doc for doc, _ in resultados[:k]]
    hits = sum(1 for d in retrieved if d in relevantes)
    return hits / k


def recall_at_k(resultados, relevantes, k=10):
    retrieved = [doc for doc, _ in resultados[:k]]
    hits = sum(1 for d in retrieved if d in relevantes)
    return hits / len(relevantes) if relevantes else 0


def average_precision(resultados, relevantes, k=10):
    retrieved = [doc for doc, _ in resultados[:k]]
    precisions = []
    hits = 0

    for i, doc in enumerate(retrieved, start=1):
        if doc in relevantes:
            hits += 1
            precisions.append(hits / i)

    return sum(precisions) / len(relevantes) if relevantes else 0


def evaluar_queries(model, df):
    print("\n EJECUTANDO EVALUACIÓN ")

    with open(QRELS_PATH, "r") as f:
        qrels = json.load(f)

    queries = {
        "q1": "open world rpg",
        "q2": "horror atmosphere",
        "q3": "multiplayer shooting"
    }

    metrics = {
        "JACCARD": [],
        "TFIDF": [],
        "BM25": []
    }

    for qname, qtext in queries.items():
        print(f"\n>>> Query {qname}: \"{qtext}\"")

        relevantes = set(qrels[qname])
        tokens_q = preprocesar(qtext)

        # Ejecutar modelos
        res_j = model.consulta_jaccard(tokens_q)
        res_t = model.consulta_tfidf(tokens_q)
        res_b = model.consulta_bm25(tokens_q)

        # Guardar métricas
        for nombre, resultados in [
            ("JACCARD", res_j),
            ("TFIDF", res_t),
            ("BM25", res_b)
        ]:
            p = precision_at_k(resultados, relevantes)
            r = recall_at_k(resultados, relevantes)
            ap = average_precision(resultados, relevantes)

            metrics[nombre].append(ap)

            print(f"\n  Modelo: {nombre}")
            print(f"    Precision@10: {p:.4f}")
            print(f"    Recall@10:    {r:.4f}")
            print(f"    AP:           {ap:.4f}")

    print("\n==== RESULTADO FINAL (MAP) ====")
    for modelo in metrics:
        map_score = sum(metrics[modelo]) / len(metrics[modelo])
        print(f"{modelo} → MAP = {map_score:.4f}")

    print("\nEvaluación completada.\n")



def main():
    df = cargar_y_preparar()
    print(f"Corpus cargado. Documentos: {len(df)}")

    print("Construyendo índice invertido...")
    indice = construir_indice(df)
    print("Índice invertido construido.\n")

    model = IRModelIndex(indice, df.index.tolist())

    print("\n SISTEMA DE RI - REVIEWS STEAM")
    print("Escribe una consulta o 'evaluar' para correr las métricas.")
    print("Escribe 'salir' para terminar.")

    while True:
        query = input("\nConsulta > ").strip()

        if query.lower() == "salir":
            print("Adiós")
            break

        if query.lower() == "evaluar":
            evaluar_queries(model, df)
            continue

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

        # BM25
        print("\n TOP 10 BM25")
        res_b = model.consulta_bm25(tokens_q)
        for r, (doc, score) in enumerate(res_b, start=1):
            texto_preview = preview_text(df.loc[doc, "review_text"])
            print(f"\n{r}. Documento {doc} | Score={score:.4f}")
            print(f"   Texto: {texto_preview}")


if __name__ == "__main__":
    main()
