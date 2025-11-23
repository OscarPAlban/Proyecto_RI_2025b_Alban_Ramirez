import pandas as pd

def construir_indice(df_tokens_preprocesados):

    df_tokens = (
        df_tokens_preprocesados[['tokens_finales', 'doc_id']]
        .explode('tokens_finales')
        .rename(columns={'tokens_finales': 'termino'})
    )

    df_tokens.dropna(subset=['termino'], inplace=True)

    # 2. Calcular la frecuencia (TF)
    df_frecuencias = (
        df_tokens.groupby(['termino', 'doc_id'])
        .size()
        .reset_index(name='frecuencia')
    )

    # 3. Construir el diccionario final
    indice_invertido = (
        df_frecuencias.groupby('termino')[['doc_id', 'frecuencia']]
        .apply(lambda x: list(map(tuple, x.values))) 
        .to_dict()
    )
    
    return indice_invertido