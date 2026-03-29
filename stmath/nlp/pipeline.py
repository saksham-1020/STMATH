from .tokenizer import Tokenizer
from .vectorizer import Vectorizer

class NLPPipeline:

    def process(self, corpus):
        tokens = [Tokenizer.tokenize(doc) for doc in corpus]
        texts = [" ".join(t) for t in tokens]
        return Vectorizer.tfidf(texts)