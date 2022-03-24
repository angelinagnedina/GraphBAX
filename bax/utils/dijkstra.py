#!/usr/bin/env python
# coding: utf-8



from sortedcontainers import SortedList
import numpy as np




class Dijkstra:
    def __init__(self, ver):
        self.ver = ver

    def value(self, v, neigh):
        ind_1, ind_2 = v.ind, neigh.ind
        return self.cost_func[(ind_1, ind_2)] if ind_1 < ind_2 else self.cost_func[(ind_2, ind_1)] 

    def run_alg(self, start, end, cost_func):
        self.cost_func = cost_func
        self.start = self.ver[start]
        self.end = self.ver[end]
        n = len(self.ver)
        dist = [float("inf") for _ in range(n)]
        prev = [None for _ in range(n)]
        visited = [False for _ in range(n)]
        s = SortedList()
        dist[start] = 0
        s.add((dist[start], self.start))
        while len(s) != 0:
            d, v = s.pop(0)
            visited[v.ind] = True
            for neigh in v.neighbors:
                if visited[neigh.ind] == False:
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
            v_1, v_2 = self.ver[ver_in_path], self.ver[prev[ver_in_path]]
            # Here we restrict situations with diagonally crossing edges (it will cause problems to computing inverse matrix)
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
