from .layers import MLP

def simple_mlp(input_size):
    return MLP(input_size, [4, 4, 1])