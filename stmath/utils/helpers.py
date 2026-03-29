def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]