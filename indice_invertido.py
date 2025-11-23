import pandas as pd

def construir_indice(df_tokens_preprocesados):


    df_tokens = df_tokens_preprocesados.copy()
    df_tokens["id"] = df_tokens.index

    df_tokens = (
        df_tokens[["id", "tokens"]]
        .explode("tokens")
        .rename(columns={"tokens": "termino"})
    )

    df_tokens.dropna(subset=["termino"], inplace=True)

    # Calcular la frecuencia TF por término y documento
    df_frecuencias = (
        df_tokens.groupby(["termino", "id"])
        .size()
        .reset_index(name="frecuencia")
    )

    indice_invertido = (
        df_frecuencias.groupby("termino")[["id", "frecuencia"]]
        .apply(lambda x: list(map(tuple, x.values)))
        .to_dict()
    )

    return indice_invertido
