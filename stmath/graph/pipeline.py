from .graph import Graph, GraphAPI

class GraphPipeline:

    def __init__(self):
        self.graph = Graph()

    def add_edge(self, u, v, w=1):
        self.graph.add_edge(u, v, w)

    def shortest_path(self, start, end):
        adj = self.graph.get_graph()
        return GraphAPI.shortest_path(adj, start, end)