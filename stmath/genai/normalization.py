def layer_norm(x):
    mean = sum(x)/len(x)
    var = sum((xi-mean)**2 for xi in x)/len(x)
    std = (var + 1e-6)**0.5
    return [(xi-mean)/std for xi in x]