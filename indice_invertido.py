def construir_indice(df_tokens_preprocesados):

    df_tokens = df_tokens_preprocesados.reset_index().copy()
 
    df_tokens = (
        df_tokens[["doc_id", "tokens"]]
        .explode("tokens")
        .rename(columns={"tokens": "termino"})
    )

    df_tokens.dropna(subset=["termino"], inplace=True)

    df_frecuencias = (
        df_tokens.groupby(["termino", "doc_id"])
        .size()
        .reset_index(name="frecuencia")
    )

    indice_invertido = (
        df_frecuencias
        .groupby("termino")[["doc_id", "frecuencia"]]
        .apply(lambda x: list(map(tuple, x.values)))
        .to_dict()
    )

    return indice_invertido
