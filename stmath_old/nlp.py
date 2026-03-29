# import math
# from typing import List, Dict, Set
# from .special import sqrt_custom, fast_ln

# class Tokenizer:
#     @staticmethod
#     def clean_and_tokenize(text: str) -> List[str]:
#         """Custom tokenizer: Removes punctuation and lowercases without regex."""
#         punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
#         clean_text = "".join(char.lower() for char in text if char not in punctuation)
#         return clean_text.split()

#     @staticmethod
#     def get_ngrams(tokens: List[str], n: int = 2) -> List[tuple]:
#         """Generates N-Grams (Bi-grams, Tri-grams) for context analysis."""
#         return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

# class Vectorizer:
#     @staticmethod
#     def term_frequency(tokens: List[str]) -> Dict[str, float]:
#         freq = {}
#         for t in tokens:
#             freq[t] = freq.get(t, 0) + 1
#         total = len(tokens)
#         return {word: count / total for word, count in freq.items()}

#     @staticmethod
#     def tf_idf(corpus: List[str]) -> List[Dict[str, float]]:
#         """
#         TF-IDF implementation from scratch. 
#         Used to rank word importance in a document set.
#         """
#         tokenized_corpus = [Tokenizer.clean_and_tokenize(doc) for doc in corpus]
#         num_docs = len(corpus)
        
#         # Calculate IDF
#         word_doc_count = {}
#         for doc_tokens in tokenized_corpus:
#             unique_words = set(doc_tokens)
#             for word in unique_words:
#                 word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
#         idf = {word: fast_ln(num_docs / count) for word, count in word_doc_count.items()}
        
#         # Calculate TF-IDF for each document
#         tfidf_matrix = []
#         for doc_tokens in tokenized_corpus:
#             tf = Vectorizer.term_frequency(doc_tokens)
#             tfidf_matrix.append({word: val * idf[word] for word, val in tf.items()})
#         return tfidf_matrix

# class NLPSimilarity:
#     @staticmethod
#     def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
#         """
#         Measures the cosine of the angle between two text vectors.
#         Fundamental for Plagiarism detection and Semantic search.
#         """
#         all_words = set(vec1.keys()) | set(vec2.keys())
        
#         dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
        
#         mag1 = sqrt_custom(sum(val**2 for val in vec1.values()))
#         mag2 = sqrt_custom(sum(val**2 for val in vec2.values()))
        
#         if mag1 == 0 or mag2 == 0: return 0.0
#         return dot_product / (mag1 * mag2)