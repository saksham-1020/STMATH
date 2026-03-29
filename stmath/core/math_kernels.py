def sqrt(x, iterations=20):
    if x < 0:
        raise ValueError("Negative sqrt undefined")
    if x == 0:
        return 0.0

    res = x if x > 1 else 1.0  # better initial guess

    for _ in range(iterations):
        res = 0.5 * (res + x / res)

    return res


def exp(x, terms=30):
    res = 1.0
    term = 1.0

    for i in range(1, terms):
        term *= x / i
        res += term

    return res


def log(x, iterations=50):
    if x <= 0:
        raise ValueError("log undefined")

    # better initial guess
    y = x - 1.0

    for _ in range(iterations):
        e_y = exp(y)
        y += 2 * (x - e_y) / (x + e_y)  # Halley

    return y


def sigmoid(x):
    # stable sigmoid
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        z = exp(x)
        return z / (1 + z)


def tanh(x):
    e_pos = exp(x)
    e_neg = exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)


def relu(x):
    return x if x > 0 else 0.0