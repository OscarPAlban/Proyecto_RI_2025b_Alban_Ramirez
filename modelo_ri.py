
import math
from collections import defaultdict, Counter

class IRModelIndex:
    def __init__(self, indice_invertido, df_index_list):

        self.index = indice_invertido
        self.docs = list(df_index_list)
        self.N = len(self.docs)

 
        self.doc_terms = defaultdict(set)

        self.doc_len = defaultdict(int)

        for term, posting in self.index.items():
            for doc_id, tf in posting:
                self.doc_terms[doc_id].add(term)
                self.doc_len[doc_id] += int(tf)

  
        total_len = sum(self.doc_len.get(d, 0) for d in self.docs)
        self.avgdl = (total_len / self.N) if self.N > 0 else 0.0

    # JACCARD 

    def consulta_jaccard(self, tokens_query, top_k=10):
        q_set = set(tokens_query)
        resultados = []
        for d in self.docs:
            terms_d = self.doc_terms.get(d, set())
            inter = len(q_set & terms_d)
            union = len(q_set | terms_d)
            score = (inter / union) if union > 0 else 0.0
            resultados.append((d, score))
        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_k]

    # TF-IDF 

    def _df(self, term):
        return len(self.index.get(term, []))

    def _tf_in_doc(self, term, doc_id):
        posting = self.index.get(term, [])
        for d, tf in posting:
            if d == doc_id:
                return int(tf)
        return 0

    def _tfidf_query_vector(self, tokens_query):
        tf_q = Counter(tokens_query)
        vec_q = {}
        for t, fq in tf_q.items():
            df = self._df(t)
            if df == 0:
                continue
            idf = math.log((self.N) / df) if df > 0 else 0.0
            w = (1 + math.log(fq)) * idf
            vec_q[t] = w
        return vec_q

    def _tfidf_doc_vector(self, doc_id):
        vec_d = {}

        for term, posting in self.index.items():
            # si el doc aparece en posting, obtener tf
            for d, tf in posting:
                if d == doc_id:
                    df = len(posting)
                    idf = math.log((self.N) / df) if df > 0 else 0.0
                    vec_d[term] = (1 + math.log(int(tf))) * idf
                    break
        return vec_d

    def _cosine_sim(self, vec_q, vec_d):
        if not vec_q or not vec_d:
            return 0.0
        inter = 0.0
        norm_q = 0.0
        norm_d = 0.0
        for t, wq in vec_q.items():
            norm_q += wq * wq
            if t in vec_d:
                inter += wq * vec_d[t]
        for _, wd in vec_d.items():
            norm_d += wd * wd
        if norm_q == 0 or norm_d == 0:
            return 0.0
        return inter / (math.sqrt(norm_q) * math.sqrt(norm_d))

    def consulta_tfidf(self, tokens_query, top_k=10):
        # Vector TFIDF de la consulta
        vec_q = self._tfidf_query_vector(tokens_query)


        candidatos = set()
        for t in tokens_query:
            for doc_id, _ in self.index.get(t, []):
                candidatos.add(doc_id)

        resultados = []

        for d in candidatos:
            vec_d = self._tfidf_doc_vector(d)
            score = self._cosine_sim(vec_q, vec_d)
            resultados.append((d, score))

        # ordenar por score descendente
        resultados.sort(key=lambda x: x[1], reverse=True)

        return resultados[:top_k]


    # BM25 

    def consulta_bm25(self, tokens_query, k1=1.5, b=0.75, top_k=10):
        resultados = []
        q_terms = tokens_query
        for d in self.docs:
            score = 0.0
            dl = self.doc_len.get(d, 0)
            for t in q_terms:
                posting = self.index.get(t, [])
                df = len(posting)
                if df == 0:
                    continue
                tf = 0
                for doc_id, f in posting:
                    if doc_id == d:
                        tf = int(f)
                        break
                if tf == 0:
                    continue
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                numer = tf * (k1 + 1.0)
                denom = tf + k1 * (1.0 - b + b * (dl / (self.avgdl if self.avgdl>0 else 1.0)))
                score += idf * (numer / denom)
            resultados.append((d, score))
        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_k]
