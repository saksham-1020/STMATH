import math

class Similarity:

    @staticmethod
    def cosine(v1, v2):
        keys = set(v1) | set(v2)

        dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
        mag1 = math.sqrt(sum(v**2 for v in v1.values()))
        mag2 = math.sqrt(sum(v**2 for v in v2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0

        return dot / (mag1 * mag2)