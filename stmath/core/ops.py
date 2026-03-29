from .value import Value

def to_value(x):
    return x if isinstance(x, Value) else Value(x)

def add(a, b):
    return to_value(a) + to_value(b)

def sub(a, b):
    return to_value(a) - to_value(b)

def mul(a, b):
    return to_value(a) * to_value(b)

def div(a, b):
    return to_value(a) / to_value(b)

def square(x):
    return to_value(x) ** 2

def relu(x):
    return to_value(x).relu()

def tanh(x):
    return to_value(x).tanh()

def exp(x):
    return to_value(x).exp()

def log(x):
    return to_value(x).log()