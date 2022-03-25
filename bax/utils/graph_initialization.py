#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection as LC




class Vertex:
    """
        This class defines the vertex with all needed attributes.
        
        Attributes:
            ind: The unique index of the vertex.
            position: Coordinates of the vertex.
            neighbors: Adjacent vertices.
    """
    
    def __init__(self, ind: int, position, neighbors = None):
        # A vertex has a unique number, its coordinates 
        # and a list of its neighbors 
        self.ind = ind 
        self.position = position
        self.neighbors = [] if neighbors is None else neighbors

    def __repr__(self):
        return f"({self.ind}, {[n.ind for n in self.neighbors]})"

    def __lt__(self, other):
        return self.position[0] < other.position[0]




def make_ver(is_edge, pos):
    """
        Args:
            is_edge: The adjacent matrix of the graph.
            pos: Positions of the vertices.
        
        Returns:
            Vertices that are instances of the class Vertex.
    """
    
    ver = [Vertex(i, p) for i, p in enumerate(pos)]
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if is_edge[i][j]:
                ver[i].neighbors.append(ver[j])
                ver[j].neighbors.append(ver[i])
    return ver




def make_edge(ver):
    """
        Args:
            ver: Vertices of the graph.
        
        Returns:
            Edges (each edge is a tuple ((v_1, pos_1), (v_2, pos_2)), 
            where v_1, v_2 are adjacent vertices and
            pos_1, pos_2 are their positions).
    """
    
    edges = []
    for v in ver:
        for n in v.neighbors:
            edges.append(((v.ind, v.position), (n.ind, n.position)))
    return edges




def change_edges(edges):
    """
        Args:
            edges: The output of the make_edge function.
        
        Returns:
            Edges (each edge is a tuple ((v_1, v_2), pos), 
            where v_1, v_2 are adjacent vertices and 
            pos is the coordinates of the middle point between v_1 and v_2).
    """
    
    new_edges = []
    for e in edges:
        if e[0][0] < e[1][0]:
            new_edges.append(((e[0][0], e[1][0]), 
                              np.sum([e[0][1], e[1][1]], axis = 0)/2))
    return new_edges




def set_weights(edges, new_edges, func):
    """
        Args:
            edges: The output of the make_edge function.
            new_edges: The output of the change_edges function.
            func: The cost function.
        
        Returns:
            A dictionary that contains cost of each edge of the graph.
            Keys are tuples (v_1_ind, v_2_ind), where v_1_ind, v_2_ind are
            indices of the adjacent vertices and v_1_ind < v_2_ind.
    """
    
    weights = {}
    cnt = 0
    for i in range(len(edges)):
        if edges[i][0][0] < edges[i][1][0]:
            # Make sure the function returns only non-negative weights            
            assert func(new_edges[cnt][1]) > 0
            weights[(edges[i][0][0], edges[i][1][0])] = func(new_edges[cnt][1])
            cnt += 1
    return weights




def plot_graph(ax, edges, pos):
    # plot edges
    color = (0.75, 0.75, 0.75, 0.1)
    lc = LC(edges, colors=[color] * len(edges), linewidths=1.0)
    ax.add_collection(lc)

    # plot vertices
    ax.scatter(*pos.T, color=(0, 0, 0, 1), 
               marker='.', facecolors='none', s=20)

    ax.grid(False)
    return



def rosenbrock(x, a = 1, b = 100):
    """
        An example of cost function from the paper.
    """
    
    return 0.01*((a - x[0])**2 + b*(x[1] - x[0]**2)**2) 


class MakeGrid2d:
    """
    Class for initializing a 2d grid-shaped graph.
    
    Attributes:
        num_grid: Number of vertices along x and y axes.
        x_lim, y_lim: A limited area on the plane 
          where the graph will be located. 
        func: The cost function.
        get_pos, get_ver, get_edges: Positions of the vertices, 
          vertices and edges of the graph respectively.
        weights: A dictionary that contains weights of the edges.
    """
    
    def __init__(self, num_grid = (20, 20), x_lim = [-2, 2], 
                 y_lim = [-1, 4], func = rosenbrock):
        self.num_grid = num_grid
        self.x_lim = np.array(x_lim)
        self.y_lim = np.array(y_lim)
        self.func = func
        self.get_pos, self.get_ver, self.get_edges, self.weights = self.__get__()
        
    def __get__(self):
        num_grid_x, num_grid_y = self.num_grid
        x = np.meshgrid(np.linspace(*self.x_lim, num_grid_x)
        y = np.linspace(*self.y_lim, num_grid_y))
        positions = np.stack([x.flatten(), y.flatten()], axis = -1) 
        pos_len = len(positions)
        is_there_edge = [[False for _ in range(pos_len)] for _ in range(pos_len)]
        # Connecting vertices in directions N, E, S, W, NW, NE, SW, SE
        for i in range(pos_len):
            for j in range(i + 1, pos_len):
                if (j - i == 1 and j % num_grid_x != 0) or 
                    (j == i + num_grid_x):
                    is_there_edge[i][j] = True
                elif (j == i + num_grid_x + 1 and 
                      i % num_grid_x != num_grid_x - 1) or 
                    (j == i + num_grid_x - 1 and 
                     i % num_grid_x != 0):
                    is_there_edge[i][j] = True
        is_there_edge = np.array(is_there_edge)
        vertices = make_ver(is_there_edge, positions)
        # This attribute is made for plot_ method        
        self.edges = make_edge(vertices)
        new_edges = change_edges(self.edges)
        weights = set_weights(self.edges, new_edges, self.func)

        return positions, vertices, new_edges, weights
    
    def plot_(self):
        """
            Method for visualizing the graph.
        """
        edges = [(e[0][1], e[1][1]) for e in self.edges]
        fig, ax = plt.subplots(figsize=(8.9, 7))
        plot_graph(ax, edges, self.positions)
        ax.set(ylim=self.y_lim, xlim=self.x_lim) 
        plt.show()
    




class GraphProcessing:
    """
    Class for initializing an arbitrary graph.
    
    Attributes:
        make_grid: True for initializing a 2d grid-shaped graph.
        graph: If make_grid = True, then graph is an instance of MakeGrid2d,
          else it is an instance of nx.classes.graph.Graph.
        get_pos, get_ver, get_edges: Positions of the vertices, 
          vertices and edges of the graph respectively.
        weights: A dictionary that contains weights of the edges.
    """
    
    def __init__(self, graph = None, make_grid = None):
        """
            Args:
                graph: Either None (if make_grid = True), or one of the 
                  dict/list/np.ndarray/nx.classes.graph.Graph types. 
                  If dict, the keys should be the indices of the vertices 
                  and the values should be the neighbors.
                  If list/np.ndarray, the input is an adjacent matrix. 
        """
        
        self.graph = graph
        self.make_grid = make_grid
     
    def create_graph(self, params = None):
        """
            Method that initializes a graph. 
            
            Args:
                params: A dictionary of parameter values. If
                  self.make_grid = True, it may contain attributes 
                  for an instance of MakeGrid2d class. Else it may 
                  contain the cost function as 'func' keyword (necessary 
                  if self.graph is a list/np.ndarray object). For adjacent 
                  matrix weights of the edges may be defined in cells of 
                  the matrix. 
        """
        
        if params is None:
            params = {}
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
                parameters.append(rosenbrock)
            graph = MakeGrid2d(*parameters)
            self.graph = graph
            self.get_pos = graph.get_pos
            self.get_ver = graph.get_ver
            self.get_edges = graph.get_edges
            self.weights = graph.weights
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
                if neigh:
                    for n in neigh:
                        graph.add_edge(key, n)               
        if type(G) == np.ndarray or type(G) == list:
            graph.add_nodes_from(range(len(G)))
            for i in range(len(G)):
                for j in range(i + 1, len(G)):
                    if G[i][j]:
                        if params.get('func') is None:
                            graph.add_edge(i, j, weight = G[i][j])
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
            self.get_pos = pos
            self.get_ver = ver
            self.get_edges = new_edges
            self.weights = weights 
            
    def draw(self):
        """
            Method for visualizing the graph.
        """
        
        if self.make_grid:
            self.graph.plot_()
        else:
            nx.draw(self.graph)
            
