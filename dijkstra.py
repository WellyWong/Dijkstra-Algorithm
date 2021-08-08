"""
Dijkstra is a greedy algorithm
https://bradfieldcs.com/algos/graphs/dijkstras-algorithm/  Credit: https://bradfieldcs.com/algos/
https://www.youtube.com/watch?v=2E7MmKv0Y24    MIT OpenCourseWare, 16. Dijkstra -- By Prof Srini Devadas

The algorithm we are going to use to determine the shortest path is called “Dijkstra’s algorithm.”
Dijkstra’s algorithm is an iterative algorithm that provides us with the shortest path from one particular
starting node to all other nodes in the graph. Again this is similar to the results of a breadth first search.

To keep track of the total cost from the start node to each destination we will make use of a distances dictionary
which we will initialize to 0 for the start vertex, and infinity for the other vertices. Our algorithm will update
these values until they represent the smallest weight path from the start to the vertex in question,
at which point we will return the distances dictionary.

The algorithm iterates once for every vertex in the graph; however, the order that we iterate over the vertices
is controlled by a priority queue. The value that is used to determine the order of the objects in the priority queue
is the distance from our starting vertex. By using a priority queue, we ensure that as we explore one vertex
after another, we are always exploring the one with the smallest distance.
"""
import heapq
import math

def dijkstra(graph, start):     # O(V + E log E) time
    d = {u: math.inf for u in graph}
    d[start] = 0
    pq = [(0, start)]

    while len(pq) > 0:
        current_distance, current_vertex = heapq.heappop(pq)
        #if current_distance > d[current_vertex]:
            #continue

        for v in graph[current_vertex]:
            if d[v] > current_distance + graph[current_vertex][v]:
                d[v] = current_distance + graph[current_vertex][v]
                heapq.heappush(pq, (d[v], v))
    return d

graph = {
    'U': {'V': 2, 'W': 5, 'X': 1},
    'V': {'U': 2, 'X': 2, 'W': 3},
    'W': {'V': 3, 'U': 5, 'X': 3, 'Y': 1, 'Z': 5},
    'X': {'U': 1, 'V': 2, 'W': 3, 'Y': 1},
    'Y': {'X': 1, 'W': 1, 'Z': 1},
    'Z': {'W': 5, 'Y': 1},
}

print(dijkstra(graph, 'X'))
#Result : {'U': 1, 'V': 2, 'W': 2, 'X': 0, 'Y': 1, 'Z': 2}
#Image of the graph: https://bradfieldcs.com/algos/graphs/dijkstras-algorithm/figures/route-graph.png







