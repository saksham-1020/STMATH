def simple_embedding(tokens, vocab):
    return [vocab.get(t, 0) for t in tokens]