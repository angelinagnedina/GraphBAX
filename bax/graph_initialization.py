#!/usr/bin/env python
# coding: utf-8

# # Какие опции будут доступны?

# - Задать граф словарём, где ключ - id вершины, а значение - соседи вершины. Веса задаются функцией, переданной на вход
# - Задать граф матрицей смежности, тогда id - номер вершины в столбце. Веса можно вписывать в ячейки матрицы или задать функцией
# - Дать на вход готовый граф с весами
# - Можно вызвать функцию make_grid_2d()

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection as LC


# In[2]:


class Vertex:
    def __init__(self, ind: int, position: np.array, neighbors = None):
        # A vertex has a unique number, its coordinates and a list of its neighbors 
        self.ind = ind 
        self.position = position
        self.neighbors = [] if neighbors is None else neighbors

    def __repr__(self):
        return f"({self.ind}, {[n.ind for n in self.neighbors]})"

    def __lt__(self, other):
        return self.position[0] < other.position[0]


# In[3]:


def make_ver(is_edge: np.array, pos: np.array):
    ver = [Vertex(i, p) for i, p in enumerate(pos)]
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if is_edge[i][j]:
                ver[i].neighbors.append(ver[j])
                ver[j].neighbors.append(ver[i])
    return ver


# In[4]:


def make_edge(ver):
    edges = []
    for v in ver:
        for n in v.neighbors:
            edges.append(((v.ind, v.position), (n.ind, n.position)))
    return edges


# In[5]:


def change_edges(edges):
    new_edges = []
    for e in edges:
        if e[0][0] < e[1][0]:
            new_edges.append(((e[0][0], e[1][0]), np.sum([e[0][1], e[1][1]], axis = 0)/2))
    return new_edges


# In[6]:


def set_weights(edges, new_edges, func):
    weights = {}
    cnt = 0
    for i in range(len(edges)):
        if edges[i][0][0] < edges[i][1][0]:
            assert func(new_edges[cnt][1]) > 0
            weights[(edges[i][0][0], edges[i][1][0])] = func(new_edges[cnt][1])
            cnt += 1
    return weights


# In[7]:


def plot_graph(ax, edges, pos):
    # plot edges
    color = (0.75, 0.75, 0.75, 0.1)
    lc = LC(edges, colors=[color] * len(edges), linewidths=1.0)
    ax.add_collection(lc)

    # plot vertices
    ax.scatter(*pos.T, color=(0, 0, 0, 1), marker='.', facecolors='none', s=20)

    ax.grid(False)
    return


# In[8]:

def Rosenbrock(x, a = 1, b = 100):
    return 0.01*((a - x[0])**2 + b*(x[1] - x[0]**2)**2) 

# In[9]:

class make_grid_2d:
    def __init__(self, num_grid = (20, 20), x_lim = [-2, 2], y_lim = [-1, 4], func = Rosenbrock):
        self.num_grid = num_grid
        self.x_lim = np.array(x_lim)
        self.y_lim = np.array(y_lim)
        self.func = func
        self.get_pos, self.get_ver, self.get_edges, self.weights = self.__get__()
        
    def __get__(self):
        num_grid_x, num_grid_y = self.num_grid
        x, y = np.meshgrid(np.linspace(*self.x_lim, num_grid_x), np.linspace(*self.y_lim, num_grid_y))
        self.positions = np.stack([x.flatten(), y.flatten()], axis = -1)
        is_there_edge = [[False for _ in range(len(self.positions))] for _ in range(len(self.positions))]
        # Connecting vertices in directions N, E, S, W, NW, NE, SW, SE
        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                if (j - i == 1 and j % num_grid_x != 0) or (j == i + num_grid_x):
                    is_there_edge[i][j] = True
                elif (j == i + num_grid_x + 1 and i % num_grid_x != num_grid_x - 1) or (j == i + num_grid_x - 1 and i % num_grid_x != 0):
                    is_there_edge[i][j] = True
        is_there_edge = np.array(is_there_edge)
        self.vertices = make_ver(is_there_edge, self.positions)
        self.edges = make_edge(self.vertices)
        new_edges = change_edges(self.edges)
        weights = set_weights(self.edges, new_edges, self.func)

        return self.positions, self.vertices, new_edges, weights
    
    def plot_(self):
        edges = [(e[0][1], e[1][1]) for e in self.edges]
        fig, ax = plt.subplots(figsize=(8.9, 7))
        plot_graph(ax, edges, self.positions)
        ax.set(ylim=self.y_lim, xlim=self.x_lim) 
        plt.show()
    
    def help(self):
        print(f'num_grid is the size of your grid,\nx_lim and y_lim are limits within which the graph will exists,\nfunc should return weights of the edges')


# In[10]:


class graph_processing:
    def __init__(self, graph = None, make_grid = False):
        self.graph = graph
        self.make_grid = make_grid
     
    def create_graph(self, params = {}):
        G = self.graph
        make_grid = self.make_grid
        if make_grid:
            parameters = []
            if params.get('size') is not None:
                parameters.append(params['size'])
            else:
                parameters.append((20, 20))
            if params.get('x_lim') is not None:
                parameters.append(params['x_lim'])
            else: 
                parameters.append([-2, 2])
            if params.get('y_lim') is not None:
                parameters.append(params['y_lim'])
            else:
                parameters.append([-1, 4])
            if params.get('func') is not None:
                parameters.append(params['func'])
            else:
                parameters.append(Rosenbrock)
            graph = make_grid_2d(*parameters)
            self.graph = graph
            self.get_pos, self.get_ver, self.get_edges, self.weights = graph.get_pos, graph.get_ver, graph.get_edges, graph.weights
        else:
            try:
                assert G is not None
            except:
                print('Give some graph, please')
            graph = nx.Graph()
        if type(G) == dict:
            for key in G.keys():
                graph.add_node(key)
            for key, neigh in G.items():
                if len(neigh) != 0:
                    for n in neigh:
                        graph.add_edge(key, n)               
        if type(G) == np.ndarray or type(G) == list:
            graph.add_nodes_from(range(len(G)))
            for i in range(len(G)):
                for j in range(i + 1, len(G)):
                    if G[i][j]:
                        if params['func'] is None:
                            graph.add_edge(i, j, G[i][j])
                        else:
                            graph.add_edge(i, j)
        if type(G) == nx.classes.graph.Graph:
            graph = nx.Graph(G)
            
        if not make_grid:  
            d = nx.circular_layout(graph)
            pos = [p for p in d.items()]
            ver = [Vertex(key, p) for key, p in d.items()]
            for v_1 in ver:
                for v_2 in ver:
                    if graph.has_edge(v_1.ind, v_2.ind):
                        v_1.neighbors.append(v_2)
            edges = make_edge(ver)
            new_edges = change_edges(edges)
            if params.get('func') is not None:
                weights = set_weights(edges, new_edges, params['func'])
            else:
                try:
                    assert type(G) != dict
                except:
                    print('You have to give a function for graph edges')
                weights = {}
                for e in graph.edges:
                    weights[e] = graph.get_edge_data(*e)['weight']
            self.graph = graph
            self.get_pos, self.get_ver, self.get_edges, self.weights = pos, ver, new_edges, weights 
            
    def draw(self):
        if self.make_grid:
            self.graph.plot_()
        else:
            nx.draw(self.graph)
            
    def help(self):
        print(f'If make_grid = True, then you may call create_graph\(\) method with dictionary of parameters, such as:\nsize,\nx_lim,\ny_lim,\nfunc ')
        print('If make_grid = False, then there are multiple ways to create a graph: by dict/adjacency matrix/networkx graph')
        print(f'If type\(graph\) == dict, then params[\'func\'] should be a function that returns the weights of the edges')
        print('If you select adjacency matrix/networkx graph, then params[\'func\'] can be None and instead the weights should be in matrix/graph already')

