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
import copy

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

#https://github.com/saisunku/CLRS-Python-Implementations/blob/master/shortest_paths.py

class Node:
    def __init__(self):
        self.num = -1               #Vertex index
        self.adj = []               #Adjacency list of integers that correspond to adjacent nodes
        self.weight = []            #List of weights for the edges in the same order as adjacency list
        self.key = math.inf         #Key for priority heap, it's the shortest path estimate
        self.pred = None            #Predecessor node

    def __eq__(self, other):        #Equality method needed for comparing nodes after deep copy
        return self.num == other.num

    def __repr__(self):
        return str(self.num)

class Graph:
    def __init__(self, nodes, root):
        self.root = root
        self.nodes = nodes      #List of adjacent nodes
        for idx, cur_node in enumerate(nodes):
            cur_node.num = idx

    def get_keys(self):
        keys = []
        for node in self.nodes:
            keys.append(node.key)
        return keys

#Min priority queue of nodes that operates on a key that can be arbitrarily set
class MinPriorityQ:
    def __init__(self, graph_nodes):
        self.heap = copy.deepcopy(graph_nodes)  #need to have own copy so that the graph nodes are unaffected by extract min
        self.heap_size = len(self.heap)
        print('heap size: ', self.heap_size)
        for j in range(math.floor(self.heap_size/2)-1, -1, -1):
            self.min_heapify(j)

    def __repr__(self):
        return str(self.heap)

    def get_nums(self):
        nums = []
        for node in self.heap:
            nums.append(node.num)
        return nums

    def left(self, k):
        return (k << 1) + 1

    def right(self, k):
        return (k << 1) + 2

    def parent(self, k):
        return math.ceil(k/2) - 1

    def min_heapify(self, k):
        smallest = k
        if self.left(k) < self.heap_size and self.heap[self.left(k)].key < self.heap[smallest].key:
            smallest = self.left(k)

        if self.right(k) < self.heap_size and self.heap[self.right(k)].key < self.heap[smallest].key:
            smallest = self.right(k)

        if smallest != k:
            # print('swapping')
            tmp = self.heap[k]
            self.heap[k] = self.heap[smallest]
            self.heap[smallest] = tmp

            self.min_heapify(smallest)

    def extract_min(self):
        to_return = self.heap[0]

        self.heap[0] = self.heap[self.heap_size-1]
        del self.heap[self.heap_size-1]
        self.heap_size -= 1
        self.min_heapify(0)

        return to_return

    def decrease_key(self, node_num, new_val):
        #Decrease the key value of the node with number: node_num to a new_val
        #Need to use node_num to identify the node here because the ordering of the node in the graph and heap are not the same
        k = None
        for idx, node in enumerate(self.heap):
            if node.num == node_num:
                k = idx
                break

        assert new_val < self.heap[k].key

        self.heap[k].key = new_val
        while self.parent(k) > -1 and self.heap[self.parent(k)].key > new_val:
            temp = self.heap[self.parent(k)]
            self.heap[self.parent(k)] = self.heap[k]
            self.heap[k] = temp
            k = self.parent(k)


def dijkstra_clrs(graph, root=None):
    if root == None:
        root = graph.root

    root.key = 0        #set shortest path estimate of the root (start) to be 0
    heap = MinPriorityQ(graph.nodes)
    while heap.heap_size > 0:
        cur_node = heap.extract_min()
        for node, weight in zip(cur_node.adj, cur_node.weight):
            u = cur_node
            v = graph.nodes[node]
            if v.key > u.key + weight:      #relaxation
                v.key = u.key + weight
                v.pred = u
                heap.decrease_key(v.num, u.key + weight)


#graph from clrs page 659
n0 = Node()     #node s
n1 = Node()     #node t
n2 = Node()     #node x
n3 = Node()     #node y
n4 = Node()     #node z
n0.adj = [1, 3]
n0.weight = [10, 5]

n1.adj = [2, 3]
n1.weight = [1, 2]

n2.adj = [4]
n2.weight = [4]

n3.adj = [1, 2, 4]
n3.weight = [3, 9, 2]

n4.adj = [0, 2]
n4.weight = [7, 6]

G = Graph([n0, n1, n2, n3, n4], n0)
dijkstra_clrs(G)
print(G.get_keys())





