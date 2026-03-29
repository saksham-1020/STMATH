from .attention import Attention

class MultiHeadAttention:

    def __init__(self, num_heads):
        self.num_heads = num_heads

    def forward(self, q, k, v):
        heads = []

        for _ in range(self.num_heads):
            head = Attention.scaled_dot(q, k, v)
            heads.append(head)

        # concat heads
        output = []
        for i in range(len(heads[0])):
            row = []
            for h in heads:
                row.extend(h[i])
            output.append(row)

        return output