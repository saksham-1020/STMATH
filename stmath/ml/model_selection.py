def train_test_split(X, y, test_size=0.2):
    import random
    data = list(zip(X, y))
    random.shuffle(data)
    split = int(len(data)*(1-test_size))
    train, test = data[:split], data[split:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return list(X_train), list(X_test), list(y_train), list(y_test)