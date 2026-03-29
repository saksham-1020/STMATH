def trace_graph(v, nodes=None, edges=None):

    if nodes is None:
        nodes = set()
    if edges is None:
        edges = []

    nodes.add(v)

    for child in v._prev:
        edges.append((child, v))
        trace_graph(child, nodes, edges)

    return nodes, edges