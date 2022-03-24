#!/usr/bin/env python
# coding: utf-8



import numpy as np
import tensorflow as tf
import gpflow
from gpflow.config import default_float
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection as LC
import sys
sys.path.insert(0, './GPflowSampling')
from gpflow_sampling.models import PathwiseGPR
from Dijkstra import Dijkstra
from graph_initialization import Rosenbrock

np.random.seed(7)
tf.random.set_seed(7)
gpflow.config.set_default_float(np.float64)




def inv_soft(x):
    return np.log(np.exp(x) - 1)

def softplus(x):
    return np.log(np.exp(x) + 1)

def true_func(x, func = lambda x: x):
    return inv_soft(func(x))

def entropy(model, X, compute_cov = True, noise = 0):
    num_ = X.shape[0]

    # Covariance matrix determinant
    if compute_cov is True:
        # if there's a model
        det = model.predict_f(X)[1].numpy().reshape(num_) + noise
    else:
        # if entropy was called from entropy_of_post_pred_distr()
        det = [model[i][i] + noise for i in range(num_)]

    for i in range(num_):
        assert det[i] > 0
    H = 1/2 + 1/2*np.log(2*np.pi) + [np.log(det[i])/2 for i in range(num_)]
    return H
    
# For each sample from num_samples contains the shortest path and values on the edges of this path
def sample(model, alg, num_samples, edges, start = 380, finish = 399):
    paths = model.generate_paths(num_samples=num_samples, num_bases=1024)
    model.set_paths(paths)
    pos_of_edges = [e[1] for e in edges]
    samples = model.predict_f_samples(tf.constant(pos_of_edges)).numpy().reshape((num_samples, -1))
    sub_pathes = dict([(i, []) for i in range(num_samples)])
    values = dict([(i, []) for i in range(num_samples)])

    for i in range(num_samples):
        func = dict([(edges[j][0], softplus(samples[i][j])) for j in range(len(edges))])
        dist_, path_, val_ = alg.run_alg(start, finish, func)
        sub_pathes[i] = np.array(path_)
        values[i] = np.array(val_)

    return sub_pathes, values

def entropy_of_post_pred_distr(X_new, ker, data, subseq, noise):
    data = (data[0].numpy(), data[1].numpy())
    arr = np.concatenate((data[0], subseq))

    k_T = ker(X_new, arr).numpy()
    k_ = k_T.T

    I = np.zeros((len(arr), len(arr)))
    for i in range(len(data[0])):
        I[i][i] = noise

    K = ker(arr, arr).numpy()
    K_inv = np.linalg.inv(K + I)

    cov = ker(X_new, X_new).numpy() - k_T@K_inv@k_ + (noise)**2*np.eye(X_new.shape[0])

    return entropy(cov, X_new, compute_cov = False, noise = noise)




class EIG_BAX(object):
    def __init__(self, X, data, model, alg, num_sam, edges, noise = 0, start = 380, finish = 399):
        self.X = X
        self.data = data
        self.alg = alg
        self.model = model
        self.num_sam = num_sam
        self.edges = edges
        self.noise = noise
        self.start = start
        self.finish = finish
    
    def calculate(self):
        first = entropy(self.model, self.X, noise = self.noise)
        sub_s, values = sample(self.model, self.alg, self.num_sam, self.edges, self.start, self.finish)
        second = np.sum([entropy_of_post_pred_distr(self.X, self.model.kernel, self.data, sub_s[i], self.noise) for i in range(self.num_sam)], axis = 0)/self.num_sam
        return first + second

class Random_BAX:
    def __init__(self, X):
        self.X = X
        
    def calculate(self):
        ind = np.random.choice(range(len(self.X)), 1)
        result = [0 if i != ind else 1 for i in range(len(self.X))]
        return result

class Var_BAX:
    def __init__(self, X, model):
        self.X = X
        self.model = model
        
    def calculate(self):
        var = self.model.predict_f(self.X)[1].numpy()
        return var
    
def edges_of_path(path):
    edges = []
    for i in range(len(path) - 1):
        edges.append((path[i], path[i + 1]))
    return np.array(edges)

def positions_of_path(path):
    positions = [v[0][1] for v in path]
    positions.append(path[-1][1][1])
    return np.stack(positions)

def plot_contourf(fig, ax, x1_lims, x2_lims):
    x, y = np.meshgrid(np.linspace(*x1_lims), np.linspace(*x2_lims))
    cs = ax.contourf(x, y, Rosenbrock((x, y)), colors = ['#F5F1F8', '#F1E1FC', '#F0C5EE', '#E4BBE2', '#C9A5C8', '#A386A2', '#786277'])
    cbar = fig.colorbar(cs, ax=ax)

def plot_graph(ax, pos, edges, start, goal):
    # plot edges
    color = (0.75, 0.75, 0.75, 0.1)
    lc = LC(edges, colors=[color] * len(edges), linewidths=1.0)
    ax.add_collection(lc)

    # plot vertices
    ax.scatter(*pos.T, color=(0, 0, 0, 1), marker='.', facecolors='none', s=20)

    # plot start and goal vertices
    ax.scatter(*start.position, color='#FF530A', marker='s', label="Start", s=150)
    ax.scatter(*goal.position, color='#21FF65', marker='s', label="Goal", s=150)

    ax.grid(False)
    return

def plot_path(
    ax,
    path,
    path_color=(0, 0, 0, 1.),
    linewidths=2,
    linestyle="dotted",
    plot_vertices=False,
    label=None,
):
    # plot path taken
    path_lines = edges_of_path(path)
    path_lc = LC(
        path_lines,
        colors=[path_color] * len(path_lines),
        linewidths=linewidths,
        linestyle=linestyle,
        label=label,
    )
    ax.add_collection(path_lc)

    # plot visited vertices
    if plot_vertices:
        ax.scatter(*positions_of_path(path).T, color=(0, 0, 0, 1))
    return

class procedure:
    def __init__(self, budget, num_samples, start, finish, graph, gp_params, kernels, init = 1):
        # Number of requests to black-box function
        self.budget = budget
        # Number of samples for Monte Carlo
        self.num_samples = num_samples
        # Index of vertex from where to start
        self.start = start
        # Index of the vertex where to finish
        self.finish = finish
        # Info about the graph, such as: positions of vertices, vertices (each contains its own index and position),
        # edges (in the form ((v_1.ind, v_2.ind), edge_coordinates), contains only edges where v_1.ind < v_2.ind), 
        # weights of the edges (in the form ((v_1.ind, v_2.ind), weight)
        self.pos, self.ver, self.edges, self.weights = graph.get_pos, graph.get_ver, graph.get_edges, graph.weights
        self.latent_weights = {key: true_func(value) for key, value in self.weights.items()}

        #Dijkstra
        self.alg = Dijkstra(self.ver)
        # Graph
        self.graph = graph

        # randomly choose first init edges
        ind_of_first_edges = np.random.choice(range(len(self.edges)), init)
        first_edges = [self.edges[i] for i in ind_of_first_edges]

        # Indices of edges which cost we've already have
        self.already_known = set(ind_of_first_edges)
        
        # GP_params
        self.gp_params = gp_params
        self.kernels = kernels

        # Initial dataset
        self.data = (tf.constant([e[1] for e in first_edges]), tf.constant([[self.latent_weights[e[0]]] for e in first_edges]))
            
        # Our model
        self.model = PathwiseGPR(data = self.data, kernel = kernels[gp_params["kernel"]], noise_variance=gp_params["noise"])
        
    def run(self, method = 'EIG', num = 100, is_example = False, one_image = False, file_path = f'images/'):
        self.true_path = self.alg.run_alg(self.start, self.finish, self.weights)[1]
        for step in range(self.budget):
            # Randomnly choosing edges among which to select the best variant
            sub_ed_first_edition = np.random.choice(range(len(self.edges)), num)
            sub_ed = []
            for i in sub_ed_first_edition:
                if i not in self.already_known:
                    sub_ed.append(i)
            X = np.array([self.edges[i][1] for i in sub_ed])
            # Choosing acquisition function
            if method == 'EIG':
                acq_func = EIG_BAX(X, self.data, self.model, self.alg, self.num_samples, self.edges, self.gp_params["noise"], self.start, self.finish)
            if method == 'Random':
                acq_func = Random_BAX(X)
            if method == 'Var':
                acq_func = Var_BAX(X, self.model)
            ind = np.argmax(acq_func.calculate()) 
            new_place_for_request = self.edges[sub_ed[ind]]
            new_val = self.latent_weights[new_place_for_request[0]]
            self.already_known.add(sub_ed[ind])
            # print('updating dataset')
            self.data = (tf.concat((self.data[0], [new_place_for_request[1]]), 0), tf.concat((self.data[1], [[new_val]]), 0))
            self.model = PathwiseGPR(data = self.data, kernel = self.kernels[self.gp_params["kernel"]], noise_variance=self.gp_params["noise"])
            
            # Make and save image/images
            if is_example:
                if step % 20 == 0:
                    print(f'Finished iter i = {step}')
                self.paint(step, one_image, file_path)                
    
    def paint(self, step, one_image, file_path):
        edges_ = [(e[0][1], e[1][1]) for e in self.graph.edges]
        start, finish = self.ver[self.start], self.ver[self.finish]
        sampled_paths, val = sample(self.model, self.alg, 15, self.edges, self.start, self.finish)
        sampled_paths = [sampled_paths[j] for j in range(15)]

        fig, ax = plt.subplots(figsize=(8.9, 7))
        plot_contourf(fig, ax, (-2, 2), (-1, 4))
        
        plot_graph(ax, self.pos, edges_, start, finish)

        plot_path(
            ax,
            self.true_path,
            path_color=(0.2, 0.2, 0.2, 1),
            linewidths=2,
            linestyle='--',
            label='True shortest path',
        )
                    
        
        data_ = (self.data[0].numpy(), self.data[1].numpy())

        for x in data_[0][:-1]:
                ax.scatter(x[0], x[1], color=(0, 0, 0, 1), s=20)

        ax.scatter(
            data_[0][-1][0],
            data_[0][-1][1],
            color='#3F15E8',
            s=60,
            label='Next query',
        )
        
        weight = 0.1 
        for path in sampled_paths:
            plot_path(ax, path, path_color=(0, 0, 1, weight), linewidths=2, linestyle="-")

        ax.set(ylim=[-1.2, 4.2], xlim=[-2.2, 2.2]) 

        # Plot title
        ax.set_title("InfoBAX with Dijkstra's Algorithm")

        # Turn off ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if one_image:
            fig.savefig(file_path + f'image_Bax.png', bbox_inches='tight')
        else:
            fig.savefig(file_path + f'image_Bax_{step}.png', bbox_inches='tight')
        plt.close()
        
