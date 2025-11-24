import numpy as np
from scipy.sparse import lil_matrix

# ==========================================================
# 1. MATRIZ DOCUMENTO-TÉRMINO SPARSE
# ==========================================================
def construir_matriz_binaria(df):
    """
    Construye matriz binaria sparse (LIL) para Jaccard.
    Mucho más eficiente en memoria.

    df["tokens"] ya debe existir.
    """
    
    # Crear vocabulario
    vocabulario = sorted(set(t for lista in df["tokens"] for t in lista))
    vocab_index = {t: i for i, t in enumerate(vocabulario)}

    n_docs = len(df)
    n_terms = len(vocabulario)

    # Sparse matrix instead of dense numpy array
    matriz = lil_matrix((n_docs, n_terms), dtype=np.uint8)

    for fila, tokens in enumerate(df["tokens"]):
        for t in tokens:
            matriz[fila, vocab_index[t]] = 1

    return matriz.tocsr(), vocabulario


# ==========================================================
# 2. CONSULTA JACCARD SPARSE
# ==========================================================
def consulta_jaccard(tokens_query, df, matriz, vocabulario, top_k=10):
    """
    Calcula Jaccard entre la consulta (tokens list) y los documentos.
    
    matriz debe ser CSR (rápida en multiplicación).
    """

    vocab_index = {t: i for i, t in enumerate(vocabulario)}

    # Vector binario sparse para la consulta
    q_vec = np.zeros(len(vocabulario), dtype=np.uint8)
    for t in tokens_query:
        if t in vocab_index:
            q_vec[vocab_index[t]] = 1

    resultados = []

    for i in range(matriz.shape[0]):
        d_vec = matriz[i].toarray().ravel()

        inter = np.sum((q_vec & d_vec) > 0)
        union = np.sum((q_vec | d_vec) > 0)

        score = inter / union if union > 0 else 0
        resultados.append((df.index[i], score))

    resultados.sort(key=lambda x: x[1], reverse=True)

    return resultados[:top_k]


def mostrar_resultados_jaccard(resultados, df, max_chars=800):

    if len(resultados) == 0:
        print("No se encontraron documentos.")
        return

    print("\n" + "="*110)
    print("RESULTADOS DE BÚSQUEDA — Modelo Jaccard ")
    

    for rank, (doc_id, score) in enumerate(resultados, start=1):
        texto = df.loc[doc_id, "review_text"]
        preview = texto[:max_chars].replace("\n", " ")

        print(f"#{rank} Documento : {doc_id}")
        print(f" Score Jaccard: {score:.4f}")
        print(f"  Texto:")
        print(f"  {preview}...\n")
