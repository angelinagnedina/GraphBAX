#!/usr/bin/env python
# coding: utf-8


from sortedcontainers import SortedList
import numpy as np


class Dijkstra:
    """
    Realization of Dijkstra's algorithm
    
    Attributes:
        ver: Vertices of the graph of type Vertex.
        cost_func: Function/dictionary that returns weights of the edges.
        start: Vertex from where to start searching for the shortest path.
        end: Vertex where the shortest path will end.
    """
    
    def __init__(self, ver):
        self.ver = ver

    def value(self, v, neigh):
        """
            Args:
                v: Vertex.
                neigh: Neighbor of the vertex v.
                
            Returns:
                Weight of the edge between v and neigh.
        """
        ind_1, ind_2 = v.ind, neigh.ind
        if ind_1 < ind_2:
            res = self.cost_func[(ind_1, ind_2)]
        else:
            res = self.cost_func[(ind_2, ind_1)]
        return  res

    def run_alg(self, start, end, cost_func):
        """
            Returns:
                Distance from the start vertex to the end vertex in the shortest path,
                the shortest path (positions of the edges in the path),
                values of each edge of the path.
        """
        self.cost_func = cost_func
        self.start = self.ver[start]
        self.end = self.ver[end]
        n = len(self.ver)
        dist = [float('inf') for _ in range(n)]
        prev = [None for _ in range(n)]
        visited = [False for _ in range(n)]
        s = SortedList()
        dist[start] = 0
        s.add((dist[start], self.start))
        while len(s):
            d, v = s.pop(0)
            visited[v.ind] = True
            for neigh in v.neighbors:
                if not visited[neigh.ind]:
                    val = self.value(v, neigh)
                    if dist[neigh.ind] > d + val:
                        s.discard((dist[neigh.ind], neigh))
                        dist[neigh.ind] = d + val
                        prev[neigh.ind] = v.ind
                        s.add((dist[neigh.ind], neigh))
        path = []
        val = []
        d = {}
        ver_in_path = end
        while ver_in_path != start:
            v_1 = self.ver[ver_in_path] 
            v_2 = self.ver[prev[ver_in_path]]
            # Here we restrict situations with diagonally crossing edges 
            # (it will cause problems to computing inverse matrix)
            if d.get(v_1.ind + v_2.ind) is None:
                d[v_1.ind + v_2.ind] = 1
                value = np.sum([v_1.position, v_2.position], axis = 0)/2
                path.append(value)
                if v_1.ind < v_2.ind:
                    val.append(self.cost_func[(v_1.ind, v_2.ind)])
                else:
                    val.append(self.cost_func[(v_2.ind, v_1.ind)])
            ver_in_path = prev[ver_in_path]
        return dist[end], path, val
