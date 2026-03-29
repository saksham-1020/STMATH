from ..core.math_kernels import sqrt, exp


class Attention:

    @staticmethod
    def softmax(x):
        # numerical stability
        max_x = max(x)
        exps = [exp(i - max_x) for i in x]
        s = sum(exps)
        return [e / s for e in exps]

    @staticmethod
    def scaled_dot(q, k, v):

        d_k = len(k[0])

        scores = []

        for qi in q:
            row = []
            for kj in k:
                dot = sum(a * b for a, b in zip(qi, kj))
                row.append(dot / sqrt(d_k))
            scores.append(row)

        weights = [Attention.softmax(r) for r in scores]

        output = []

        for w in weights:
            out = [0.0] * len(v[0])
            for i, weight in enumerate(w):
                for j in range(len(v[0])):
                    out[j] += weight * v[i][j]
            output.append(out)

        return output