from pre_procesamiento import preprocesar
import json

def precision(relevantes, recuperados):
    if len(recuperados) == 0:
        return 0.0
    inter = len([d for d in recuperados if d in relevantes])
    return inter / len(recuperados)

def recall(relevantes, recuperados):
    if len(relevantes) == 0:
        return 0.0
    inter = len([d for d in recuperados if d in relevantes])
    return inter / len(relevantes)

def average_precision(relevantes, recuperados):
    aciertos = 0
    suma = 0.0

    for i, doc in enumerate(recuperados, start=1):
        if doc in relevantes:
            aciertos += 1
            suma += aciertos / i

    if aciertos == 0:
        return 0.0

    return suma / aciertos

def evaluar(model, df):
    # Consultas oficiales
    consultas = {
        "q1": "call of duty",
        "q2": "horror atmosphere",
        "q3": "multiplayer shooting"
    }

    # Cargar relevancia manual
    with open("qrels.json", "r") as f:
        qrels = json.load(f)

    APs = []

    for qid, consulta in consultas.items():
        print(f"\n=== {qid}: {consulta} ===")

        tokens = preprocesar(consulta)

        # Recuperar top 20
        recuperados = [doc for doc, _ in model.consulta_bm25(tokens, top_k=20)]

        relevantes = set(qrels[qid])

        P = precision(relevantes, recuperados)
        R = recall(relevantes, recuperados)
        AP = average_precision(relevantes, recuperados)

        APs.append(AP)

        print(f"Precision: {P:.4f}")
        print(f"Recall:    {R:.4f}")
        print(f"AP:        {AP:.4f}")

    # MAP final
    MAP = sum(APs) / len(APs)

    print("\n====== RESULTADO FINAL ======")
    print(f"MAP = {MAP:.4f}")

    return MAP
