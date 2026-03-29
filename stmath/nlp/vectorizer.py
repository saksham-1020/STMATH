from collections import Counter
from ..core.math_kernels import log 


class Vectorizer:

    @staticmethod
    def _tokenize(text):
        return text.lower().split()

    @staticmethod
    def tf(text):
        words = Vectorizer._tokenize(text)
        count = Counter(words)
        total = len(words) if words else 1
        return {w: c / total for w, c in count.items()}

    @staticmethod
    def idf(corpus):
        N = len(corpus)
        idf_dict = {}

        for doc in corpus:
            words = set(Vectorizer._tokenize(doc))
            for word in words:
                idf_dict[word] = idf_dict.get(word, 0) + 1

        # 🔥 zero-wrapper log
        return {w: log((N + 1) / (c + 1)) + 1 for w, c in idf_dict.items()}

    @staticmethod
    def tfidf(corpus):
        idf_vals = Vectorizer.idf(corpus)
        tfidf_all = []

        for doc in corpus:
            tf_vals = Vectorizer.tf(doc)
            vec = {}

            for w in tf_vals:
                vec[w] = tf_vals[w] * idf_vals.get(w, 0.0)

            tfidf_all.append(vec)

        return tfidf_all

    # backward compatibility
    @staticmethod
    def tf_idf(corpus):
        return Vectorizer.tfidf(corpus)