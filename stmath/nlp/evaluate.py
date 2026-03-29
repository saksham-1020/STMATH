from .similarity import Similarity

def text_similarity(doc1_vec, doc2_vec):
    return Similarity.cosine(doc1_vec, doc2_vec)