{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Какие опции будут доступны?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Задать граф словарём, где ключ - id вершины, а значение - соседи вершины. Веса задаются функцией, переданной на вход\n",
    "- Задать граф матрицей смежности, тогда id - номер вершины в столбце. Веса можно вписывать в ячейки матрицы или задать функцией\n",
    "- Дать на вход готовый граф с весами\n",
    "- Можно вызвать функцию make_grid_2d()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import networkx as nx\n",
    "from matplotlib.collections import LineCollection as LC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Vertex:\n",
    "    def __init__(self, ind: int, position: np.array, neighbors = None):\n",
    "        # A vertex has a unique number, its coordinates and a list of its neighbors \n",
    "        self.ind = ind \n",
    "        self.position = position\n",
    "        self.neighbors = [] if neighbors is None else neighbors\n",
    "\n",
    "    def __repr__(self):\n",
    "        return f\"({self.ind}, {[n.ind for n in self.neighbors]})\"\n",
    "\n",
    "    def __lt__(self, other):\n",
    "        return self.position[0] < other.position[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def make_ver(is_edge: np.array, pos: np.array):\n",
    "    ver = [Vertex(i, p) for i, p in enumerate(pos)]\n",
    "    for i in range(pos.shape[0]):\n",
    "        for j in range(i + 1, pos.shape[0]):\n",
    "            if is_edge[i][j]:\n",
    "                ver[i].neighbors.append(ver[j])\n",
    "                ver[j].neighbors.append(ver[i])\n",
    "    return ver"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def make_edge(ver):\n",
    "    edges = []\n",
    "    for v in ver:\n",
    "        for n in v.neighbors:\n",
    "            edges.append(((v.ind, v.position), (n.ind, n.position)))\n",
    "    return edges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def change_edges(edges):\n",
    "    new_edges = []\n",
    "    for e in edges:\n",
    "        if e[0][0] < e[1][0]:\n",
    "            new_edges.append(np.sum([e[0][1], e[1][1]], axis = 0)/2)\n",
    "    return new_edges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def set_weights(edges, new_edges, func):\n",
    "    weights = {}\n",
    "    for i in range(len(new_edges)):\n",
    "        assert func(new_edges[i]) >= 0\n",
    "        weights[(edges[i][0][0], edges[i][1][0])] = func(new_edges[i])\n",
    "    return weights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_graph(ax, edges, pos):\n",
    "    # plot edges\n",
    "    color = (0.75, 0.75, 0.75, 0.1)\n",
    "    lc = LC(edges, colors=[color] * len(edges), linewidths=1.0)\n",
    "    ax.add_collection(lc)\n",
    "\n",
    "    # plot vertices\n",
    "    ax.scatter(*pos.T, color=(0, 0, 0, 1), marker='.', facecolors='none', s=20)\n",
    "\n",
    "    ax.grid(False)\n",
    "    return"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Rosenbrock(x, a = 1, b = 100):\n",
    "    return 0.01*((a - x[0])**2 + b*(x[1] - x[0]**2)**2) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [],
   "source": [
    "class make_grid_2d:\n",
    "    def __init__(self, num_grid = (20, 20), x_lim = [-2, 2], y_lim = [-1, 4], func = Rosenbrock):\n",
    "        self.num_grid = num_grid\n",
    "        self.x_lim = np.array(x_lim)\n",
    "        self.y_lim = np.array(y_lim)\n",
    "        self.func = func\n",
    "        self.get_pos, self.get_ver, self.get_edges, self.weights = self.__get__()\n",
    "        \n",
    "    def __get__(self):\n",
    "        num_grid_x, num_grid_y = self.num_grid\n",
    "        x, y = np.meshgrid(np.linspace(*self.x_lim, num_grid_x), np.linspace(*self.y_lim, num_grid_y))\n",
    "        self.positions = np.stack([x.flatten(), y.flatten()], axis = -1)\n",
    "        is_there_edge = [[False for _ in range(len(self.positions))] for _ in range(len(self.positions))]\n",
    "        # Connecting vertices in directions N, E, S, W, NW, NE, SW, SE\n",
    "        for i in range(len(self.positions)):\n",
    "            for j in range(i + 1, len(self.positions)):\n",
    "                if (j - i == 1 and j % num_grid_x != 0) or (j == i + num_grid_x):\n",
    "                    is_there_edge[i][j] = True\n",
    "                elif (j == i + num_grid_x + 1 and i % num_grid_x != num_grid_x - 1) or (j == i + num_grid_x - 1 and i % num_grid_x != 0):\n",
    "                    is_there_edge[i][j] = True\n",
    "        is_there_edge = np.array(is_there_edge)\n",
    "        self.vertices = make_ver(is_there_edge, self.positions)\n",
    "        self.edges = make_edge(self.vertices)\n",
    "        new_edges = change_edges(self.edges)\n",
    "        weights = set_weights(self.edges, new_edges, self.func)\n",
    "\n",
    "        return self.positions, self.vertices, new_edges, weights\n",
    "    \n",
    "    def plot_(self):\n",
    "        edges = [(e[0][1], e[1][1]) for e in self.edges]\n",
    "        fig, ax = plt.subplots(figsize=(8.9, 7))\n",
    "        plot_graph(ax, edges, self.positions)\n",
    "        ax.set(ylim=self.y_lim, xlim=self.x_lim) \n",
    "        plt.show()\n",
    "    \n",
    "    def help(self):\n",
    "        print(f'num_grid is the size of your grid,\\nx_lim and y_lim are limits within which the graph will exists,\\nfunc should return weights of the edges')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [],
   "source": [
    "class graph_processing():\n",
    "    def __init__(self, graph = None, make_grid = False):\n",
    "        self.graph = graph\n",
    "        self.make_grid = make_grid\n",
    "     \n",
    "    def create_graph(self, params = {}):\n",
    "        G = self.graph\n",
    "        make_grid = self.make_grid\n",
    "        if make_grid:\n",
    "            parameters = []\n",
    "            if params.get('size') is not None:\n",
    "                parameters.append(params['size'])\n",
    "            else:\n",
    "                parameters.append((20, 20))\n",
    "            if params.get('x_lim') is not None:\n",
    "                parameters.append(params['x_lim'])\n",
    "            else: \n",
    "                parameters.append([-2, 2])\n",
    "            if params.get('y_lim') is not None:\n",
    "                parameters.append(params['y_lim'])\n",
    "            else:\n",
    "                parameters.append([-1, 4])\n",
    "            if params.get('func') is not None:\n",
    "                parameters.append(params['func'])\n",
    "            else:\n",
    "                parameters.append(Rosenbrock)\n",
    "            graph = make_grid_2d(*parameters)\n",
    "            self.graph = graph\n",
    "            self.get_pos, self.get_ver, self.get_edges, self.weights = graph.get_pos, graph.get_ver, graph.get_edges, graph.weights\n",
    "        else:\n",
    "            try:\n",
    "                assert G is not None\n",
    "            except:\n",
    "                print('Give some graph, please')\n",
    "            graph = nx.Graph()\n",
    "        if type(G) == dict:\n",
    "            for key in G.keys():\n",
    "                graph.add_node(key)\n",
    "            for key, neigh in G.items():\n",
    "                if len(neigh) != 0:\n",
    "                    for n in neigh:\n",
    "                        graph.add_edge(key, n)               \n",
    "        if type(G) == np.ndarray or type(G) == list:\n",
    "            graph.add_nodes_from(range(len(G)))\n",
    "            for i in range(len(G)):\n",
    "                for j in range(i + 1, len(G)):\n",
    "                    if G[i][j]:\n",
    "                        if params['func'] is None:\n",
    "                            graph.add_edge(i, j, G[i][j])\n",
    "                        else:\n",
    "                            graph.add_edge(i, j)\n",
    "        if type(G) == nx.classes.graph.Graph:\n",
    "            graph = nx.Graph(G)\n",
    "            \n",
    "        if not make_grid:  \n",
    "            d = nx.circular_layout(graph)\n",
    "            pos = [p for p in d.items()]\n",
    "            ver = [Vertex(key, p) for key, p in d.items()]\n",
    "            for v_1 in ver:\n",
    "                for v_2 in ver:\n",
    "                    if graph.has_edge(v_1.ind, v_2.ind):\n",
    "                        v_1.neighbors.append(v_2)\n",
    "            edges = make_edge(ver)\n",
    "            new_edges = change_edges(edges)\n",
    "            if params.get('func') is not None:\n",
    "                weights = set_weights(edges, new_edges, params['func'])\n",
    "            else:\n",
    "                try:\n",
    "                    assert type(G) != dict\n",
    "                except:\n",
    "                    print('You have to give a function for graph edges')\n",
    "                weights = {}\n",
    "                for e in graph.edges:\n",
    "                    weights[e] = graph.get_edge_data(*e)['weight']\n",
    "            self.graph = graph\n",
    "            self.get_pos, self.get_ver, self.get_edges, self.weights = pos, ver, new_edges, weights \n",
    "            \n",
    "    def draw(self):\n",
    "        if self.make_grid:\n",
    "            self.graph.plot_()\n",
    "        else:\n",
    "            nx.draw(self.graph)\n",
    "            \n",
    "    def help(self):\n",
    "        print(f'If make_grid = True, then you may call create_graph\\(\\) method with dictionary of parameters, such as:\\nsize,\\nx_lim,\\ny_lim,\\nfunc ')\n",
    "        print('If make_grid = False, then there are multiple ways to create a graph: by dict/adjacency matrix/networkx graph')\n",
    "        print(f'If type\\(graph\\) == dict, then params[\\'func\\'] should be a function that returns the weights of the edges')\n",
    "        print('If you select adjacency matrix/networkx graph, then params[\\'func\\'] can be None and instead the weights should be in matrix/graph already')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
