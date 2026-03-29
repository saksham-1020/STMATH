from .embeddings import simple_embedding
from .transformer import TransformerBlock


class GenAIPipeline:

    def __init__(self):
        self.model = TransformerBlock(d_model=4)

    def build_vocab(self, tokens):
        vocab = {}
        for i, t in enumerate(set(tokens)):
            vocab[t] = i
        return vocab

    def run(self, tokens, vocab=None):

        if vocab is None:
            vocab = self.build_vocab(tokens)

        embedded = [simple_embedding(t, vocab) for t in tokens]

        output = self.model(embedded, embedded, embedded)

        return output