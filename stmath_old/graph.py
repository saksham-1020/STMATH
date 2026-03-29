# import heapq
# from collections import deque
# from .special import sqrt_custom

# class GraphEngine:
#     # --- Classic Search Algorithms ---
#     @staticmethod
#     def dfs(adj, start, visited=None):
#         """Standard DFS: Useful for cycle detection and connectivity."""
#         if visited is None: visited = set()
#         visited.add(start)
#         path = [start]
#         for neighbor in adj.get(start, []):
#             v = neighbor[0] if isinstance(neighbor, tuple) else neighbor
#             if v not in visited:
#                 path.extend(GraphEngine.dfs(adj, v, visited))
#         return path

#     @staticmethod
#     def bfs(adj, start):
#         """Breadth-First Search: Shortest path for unweighted graphs."""
#         q = deque([start])
#         dist = {start: 0}
#         visited = {start}
#         while q:
#             u = q.popleft()
#             for neighbor in adj.get(u, []):
#                 v = neighbor[0] if isinstance(neighbor, tuple) else neighbor
#                 if v not in visited:
#                     visited.add(v)
#                     dist[v] = dist[u] + 1
#                     q.append(v)
#         return dist

#     # --- Weighted Pathfinding (MNC/Maps Grade) ---
#     @staticmethod
#     def dijkstra(adj, start):
#         """Dijkstra's Algorithm: Shortest path for weighted graphs (Non-negative)."""
#         dist = {n: float("inf") for n in adj}
#         prev = {n: None for n in adj}
#         dist[start] = 0
#         pq = [(0, start)]
#         while pq:
#             d, u = heapq.heappop(pq)
#             if d > dist[u]: continue
#             for neighbor in adj.get(u, []):
#                 v, weight = neighbor if isinstance(neighbor, tuple) else (neighbor, 1)
#                 if d + weight < dist[v]:
#                     dist[v] = d + weight
#                     prev[v] = u
#                     heapq.heappush(pq, (dist[v], v))
#         return dist, prev

#     @staticmethod
#     def a_star(adj, start, goal, heuristics):
#         """A* Search: f(n) = g(n) + h(n). Industry standard for GPS and Game AI."""
#         pq = [(0 + heuristics.get(start, 0), 0, start, [])]
#         visited = set()
#         while pq:
#             (f, g, u, path) = heapq.heappop(pq)
#             if u in visited: continue
#             path = path + [u]
#             if u == goal: return path, g
#             visited.add(u)
#             for neighbor in adj.get(u, []):
#                 v, weight = neighbor if isinstance(neighbor, tuple) else (neighbor, 1)
#                 if v not in visited:
#                     new_g = g + weight
#                     new_f = new_g + heuristics.get(v, 0)
#                     heapq.heappush(pq, (new_f, new_g, v, path))
#         return None, float('inf')

#     # --- Advanced Network & AI Search ---
#     @staticmethod
#     def topological_sort(root):
#         """AI Engine Backbone: Orders nodes for backpropagation in neural nets."""
#         topo, visited = [], set()
#         def build_topo(v):
#             if v not in visited:
#                 visited.add(v)
#                 if hasattr(v, '_prev'):
#                     for child in v._prev: build_topo(child)
#                 topo.append(v)
#         build_topo(root)
#         return topo

#     @staticmethod
#     def ao_star(adj, start, weights):
#         """AO* Search: Solves AND-OR graphs for decision-making AI."""
#         def get_cost(node):
#             if node not in adj: return weights.get(node, 0)
#             costs = []
#             for conn in adj[node]:
#                 if isinstance(conn, list): # AND node
#                     costs.append(sum(get_cost(n) for n in conn) + len(conn))
#                 else: # OR node
#                     costs.append(get_cost(conn) + 1)
#             return min(costs)
#         return get_cost(start)

#     @staticmethod
#     def pagerank(adj, iterations=20, d=0.85):
#         """Google's Core Algorithm: Ranks nodes by topological importance."""
#         nodes = list(adj.keys())
#         n = len(nodes)
#         if n == 0: return {}
#         ranks = {node: 1.0/n for node in nodes}
#         for _ in range(iterations):
#             new_ranks = {node: (1-d)/n for node in nodes}
#             for node in nodes:
#                 neighbors = adj.get(node, [])
#                 if neighbors:
#                     share = d * ranks[node] / len(neighbors)
#                     for neighbor in neighbors:
#                         v = neighbor[0] if isinstance(neighbor, tuple) else neighbor
#                         if v in new_ranks: new_ranks[v] += share
#             ranks = new_ranks
#         return ranks

# class Visualizer:
#     @staticmethod
#     def trace_graph(root):
#         """Traces the computational graph for the Autograd Engine."""
#         nodes, edges = set(), set()
#         def build(v):
#             if v not in nodes:
#                 nodes.add(v)
#                 if hasattr(v, '_prev'):
#                     for child in v._prev:
#                         edges.add((child, v))
#                         build(child)
#         build(root)
#         return nodes, edges

#     @staticmethod
#     def draw_graph(root):
#         """Prints a professional trace of the math operations flow."""
#         nodes, edges = Visualizer.trace_graph(root)
#         print("\n--- STMATH Computational Trace ---")
#         for n in nodes:
#             op = f" | Op: {n._op}" if hasattr(n, '_op') and n._op else ""
#             print(f"Node Data: {n.data:.4f} | Grad: {n.grad:.4f}{op}")
#         for e in edges:
#             print(f"  {e[0].data:.2f} ---> {e[1].data:.2f}")