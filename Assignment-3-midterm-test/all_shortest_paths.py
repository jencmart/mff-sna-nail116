# Python program for the above approach

# Function to form edge between
# two vertices src and dest
from typing import List
from sys import maxsize
from collections import deque


def add_edge(adj: List[List[int]], src: int, dest: int) -> None:
    adj[src].append(dest)
    adj[dest].append(src)


# Function which finds all the paths
# and stores it in paths array
def find_paths(paths: List[List[int]], path: List[int],
               parent: List[List[int]], n: int, u: int) -> None:
    # Base Case
    if (u == -1):
        paths.append(path.copy())
        return

    # Loop for all the parents
    # of the given vertex
    for par in parent[u]:
        # Insert the current
        # vertex in path
        path.append(u)

        # Recursive call for its parent
        find_paths(paths, path, parent, n, par)

        # Remove the current vertex
        path.pop()


# Function which performs bfs
# from the given souce vertex
def bfs(adj: List[List[int]],
        parent: List[List[int]], n: int,
        start: int) -> None:
    # dist will contain shortest distance
    # from start to every other vertex
    dist = [maxsize for _ in range(n)]
    q = deque()

    # Insert source vertex in queue and make
    # its parent -1 and distance 0
    q.append(start)
    parent[start] = [-1]
    dist[start] = 0

    # Untill Queue is empty
    while q:
        u = q[0]
        q.popleft()
        for v in adj[u]:
            if (dist[v] > dist[u] + 1):

                # A shorter distance is found
                # So erase all the previous parents
                # and insert new parent u in parent[v]
                dist[v] = dist[u] + 1
                q.append(v)
                parent[v].clear()
                parent[v].append(u)

            elif (dist[v] == dist[u] + 1):

                # Another candidate parent for
                # shortes path found
                parent[v].append(u)


# Function which prints all the paths
# from start to end
def get_all_shortest_paths(adj: List[List[int]], n: int,
                           start: int, end: int) -> None:
    paths = []
    path = []
    parent = [[] for _ in range(n)]

    # Function call to bfs
    bfs(adj, parent, n, start)

    # Function call to find_paths
    find_paths(paths, path, parent, n, end)
    return paths


