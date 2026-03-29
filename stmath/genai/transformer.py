from .multihead import MultiHeadAttention
from .normalization import layer_norm
from .ffn import FFN


class TransformerBlock:

    def __init__(self, d_model=None, heads=2):
        self.d_model = d_model
        self.mha = MultiHeadAttention(heads)

    def forward(self, q, k, v):

        # attention
        attn = self.mha.forward(q, k, v)

        # residual + norm
        norm1 = [layer_norm(row) for row in attn]

        # feed forward
        ffn_out = FFN.forward(norm1)

        # second residual + norm
        output = [layer_norm(row) for row in ffn_out]

        return output

    def __call__(self, q, k, v):
        return self.forward(q, k, v)