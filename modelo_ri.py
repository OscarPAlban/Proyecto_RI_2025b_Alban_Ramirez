
import math
from collections import defaultdict, Counter

class IRModelIndex:
    def __init__(self, indice_invertido, df_index_list):
        
        self.index = indice_invertido
        self.docs = list(df_index_list)
        self.N = len(self.docs)

        self.doc_terms = defaultdict(set)
        self.doc_len = defaultdict(int)
        
        # Diccionarios para acceso rápido
        self.df = {}  # Document Frequency: df[term] = número de documentos
        self.doc_tf = defaultdict(lambda: defaultdict(int)) # Term Frequency: doc_tf[doc_id][term] = frecuencia

        # Rellenar métricas pre-calculadas
        for term, posting in self.index.items():
            self.df[term] = len(posting) 
            
            for doc_id, tf_str in posting:
                tf = int(tf_str)
                self.doc_tf[doc_id][term] = tf 
                self.doc_terms[doc_id].add(term)
                self.doc_len[doc_id] += tf

        # Calcular longitud promedio del documento
        total_len = sum(self.doc_len.get(d, 0) for d in self.docs)
        self.avgdl = (total_len / self.N) if self.N > 0 else 0.0

    # Métodos de acceso rápido

    def _df(self, term):
        return self.df.get(term, 0)

    def _tf_in_doc(self, term, doc_id):
        return self.doc_tf.get(doc_id, {}).get(term, 0)

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

    # TF-IDF (Optimizado)

    def _tfidf_query_vector(self, tokens_query):
        tf_q = Counter(tokens_query)
        vec_q = {}
        for t, fq in tf_q.items():
            df = self._df(t)
            if df == 0:
                continue
            idf = math.log(self.N / df) if df > 0 else 0.0
            w = (1 + math.log(fq)) * idf
            vec_q[t] = w
        return vec_q

    def consulta_tfidf(self, tokens_query, top_k=10):
        
        vec_q = self._tfidf_query_vector(tokens_query)
        if not vec_q:
            return []

        # Norma de la consulta
        norm_q = sum(wq * wq for wq in vec_q.values())
        norm_q_sqrt = math.sqrt(norm_q)
        
        candidatos_inter_sum = defaultdict(float) 
        doc_norms_sq = defaultdict(float)        

        # Calcular la intersección y la norma del documento (solo para términos de la consulta)
        for t, wq in vec_q.items():
            df_t = self._df(t)
            if df_t == 0: continue
            
            idf_t = math.log(self.N / df_t)
            
            for doc_id, tf_str in self.index.get(t, []):
                tf = int(tf_str)

                # Peso Wd,t del documento
                wd_t = (1 + math.log(tf)) * idf_t
                
                candidatos_inter_sum[doc_id] += wq * wd_t
                doc_norms_sq[doc_id] += wd_t * wd_t

        # Calcular el score final 
        resultados = []
        for d, inter in candidatos_inter_sum.items():
            norm_d_sq = doc_norms_sq[d]
            
            if norm_d_sq == 0 or norm_q_sqrt == 0:
                score = 0.0
            else:
                score = inter / (norm_q_sqrt * math.sqrt(norm_d_sq))
                
            resultados.append((d, score))

        resultados.sort(key=lambda x: x[1], reverse=True)

        return resultados[:top_k]

    # BM25 (Optimizado)

    def consulta_bm25(self, tokens_query, k1=1.5, b=0.75, top_k=10):
        
        q_terms = set(tokens_query) 

        # Identificar documentos candidatos
        candidatos = set()
        for t in q_terms:
            for doc_id, _ in self.index.get(t, []):
                candidatos.add(doc_id)
                
        # Calcular IDF de BM25 solo una vez
        idf_q = {}
        for t in q_terms:
            df = self._df(t)
            if df > 0:
                idf_q[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            
        resultados = []
        
        # Iterar SOLAMENTE sobre los documentos candidatos
        for d in candidatos:
            score = 0.0
            dl = self.doc_len.get(d, 0)
            
            avgdl_safe = self.avgdl if self.avgdl > 0 else 1.0
            norm_dl = (1.0 - b + b * (dl / avgdl_safe))

            for t in q_terms:
                if t not in idf_q:
                    continue

                tf = self._tf_in_doc(t, d) 
                
                if tf > 0:
                    idf = idf_q[t]
                    numer = tf * (k1 + 1.0)
                    denom = tf + k1 * norm_dl
                    score += idf * (numer / denom)
                    
            resultados.append((d, score))
            
        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_k]