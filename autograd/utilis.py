import numpy as np

# sorting graph
def topological_sort(vertices, edges):
    # Mark all the vertices as unvisited
    visited = [False]*len(vertices)
    result = []

    vertex_id_array = {v: i for i, v in enumerate(vertices)}

    # Define Helper Functions
    def get_vertex_id(vertex, vertices):
        for (i, vertex_prime) in enumerate(vertices):
            if vertices is vertex_prime:
                return i
    
    def helper(i, visited_array, result, vertices, edges):
        # We are currently vistiting vertex id
        visited_array[i] = True

        # We now visit all adjacent Vertices that have yet to be visited
        adjacent = edges[vertices[i]]
        for v in adjacent:
            new_i = vertex_id_array[v] #get_vertex_id(v, vertices)
            if not visited_array[new_i]:
                helper(new_i, visited_array, result, vertices, edges)
        
        # Place current node at the front of the result
        result.insert(0, vertices[i])

    # Perform Sort using helper function
    for i in range(len(vertices)):
        if not visited[i]:
            helper(i, visited, result, vertices, edges)
    
    return result


# generating graph
def generate_graph(vertex, vertices, edges):
    if vertex not in vertices:
        vertices.append(vertex)
        if vertex.compute_graph is not None:
            edges.update({vertex: vertex.compute_graph.vertices})
        else:
            edges.update({vertex: []})
        for v in edges[vertex]:
            generate_graph(v, vertices, edges)
