# Dijkstra-Algorithm
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
