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


def stemming(tokens):
    tokens_stemmizados = [STEMMER_PORTER.stem(palabra) for palabra in tokens]
    return tokens_stemmizados

def preprocesar(documento_texto):
    texto_limpio = limpiar(documento_texto)
    tokens_filtrados = tokenizar(texto_limpio)
    tokens_finales = stemming(tokens_filtrados)
    return tokens_finales