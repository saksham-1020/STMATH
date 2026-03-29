class Graph:
    def __init__(self):
        self.adj = {}

    def add_edge(self, u, v, weight=1):
        self.adj.setdefault(u, []).append((v, weight))
        self.adj.setdefault(v, [])

    def add_undirected_edge(self, u, v, weight=1):
        self.add_edge(u, v, weight)
        self.add_edge(v, u, weight)

    def get_graph(self):
        return self.adj



class GraphEngine:

    @staticmethod
    def dfs(adj, start):
        visited = set()
        order = []

        def _dfs(u):
            if u in visited:
                return
            visited.add(u)
            order.append(u)

            for v, _ in adj.get(u, []):
                _dfs(v)

        _dfs(start)
        return order

    @staticmethod
    def bfs(adj, start):
        visited = set([start])
        queue = [start]  # manual queue
        order = []

        while queue:
            u = queue.pop(0)  # FIFO
            order.append(u)

            for v, _ in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        return order

    @staticmethod
    def dijkstra(adj, start):
        dist = {node: float("inf") for node in adj}
        prev = {node: None for node in adj}
        visited = set()

        dist[start] = 0

        while len(visited) < len(adj):

            # 🔥 manual min extraction (no heapq)
            min_node = None
            min_dist = float("inf")

            for node in adj:
                if node not in visited and dist[node] < min_dist:
                    min_dist = dist[node]
                    min_node = node

            if min_node is None:
                break

            visited.add(min_node)

            for v, w in adj.get(min_node, []):
                if dist[min_node] + w < dist[v]:
                    dist[v] = dist[min_node] + w
                    prev[v] = min_node

        return dist, prev




def reconstruct_path(prev, start, end):
    path = []
    curr = end

    while curr is not None:
        path.append(curr)
        curr = prev[curr]

    path.reverse()

    return path if path and path[0] == start else []



class GraphAPI:

    @staticmethod
    def shortest_path(adj, start, end):
        dist, prev = GraphEngine.dijkstra(adj, start)
        path = reconstruct_path(prev, start, end)
        return path, dist[end]

    @staticmethod
    def bfs_path(adj, start, end):
        queue = [start]
        parent = {start: None}

        while queue:
            u = queue.pop(0)

            if u == end:
                break

            for v, _ in adj.get(u, []):
                if v not in parent:
                    parent[v] = u
                    queue.append(v)

        return reconstruct_path(parent, start, end)




class GraphPipeline:

    def __init__(self):
        self.graph = Graph()

    def add_edge(self, u, v, w=1):
        self.graph.add_edge(u, v, w)

    def dfs(self, start):
        return GraphEngine.dfs(self.graph.get_graph(), start)

    def bfs(self, start):
        return GraphEngine.bfs(self.graph.get_graph(), start)

    def shortest_path(self, start, end):
        return GraphAPI.shortest_path(self.graph.get_graph(), start, end)

    def bfs_path(self, start, end):
        return GraphAPI.bfs_path(self.graph.get_graph(), start, end)