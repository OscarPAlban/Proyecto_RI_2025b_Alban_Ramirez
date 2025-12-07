import re
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


STEMMER_PORTER = PorterStemmer()
PALABRAS_VACIAS = set(stopwords.words('english'))

def limpiar(documento_texto):
    documento_texto = re.sub(pattern=r'<.*?>|[(),"]', repl='', string=documento_texto)
    documento_texto = documento_texto.replace('.', ' ')
    documento_texto = documento_texto.lower()
    return documento_texto


def tokenizar(documento_texto_limpio):
    tokens = word_tokenize(documento_texto_limpio)

    tokens_filtrados = [
        palabra for palabra in tokens
        if palabra not in PALABRAS_VACIAS and palabra.isalpha()
    ]
    return tokens_filtrados

def eliminar_repetidos(tokens):
    patron = re.compile(r'^([a-zA-Z])\1+$')
    return [t for t in tokens if not patron.match(t)]

def stemming(tokens):
    tokens_stemmizados = [STEMMER_PORTER.stem(palabra) for palabra in tokens]
    return tokens_stemmizados

def generar_bigrams(tokens):
    return [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]

def preprocesar(documento_texto):
    texto_limpio = limpiar(documento_texto)
    tokens_filtrados = tokenizar(texto_limpio)
    tokens_repetidas = eliminar_repetidos(tokens_filtrados)
    tokens_finales = stemming(tokens_repetidas)

    bigrams = generar_bigrams(tokens_finales)

    tokens_finales = tokens_finales + bigrams

    return tokens_finales
