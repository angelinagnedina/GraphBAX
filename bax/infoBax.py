#!/usr/bin/env python
# coding: utf-8


import numpy as np
import tensorflow as tf
import networkx as nx
from scipy import sparse
import gpflow
from gpflow_sampling.models import PathwiseGPR
from graph_matern.kernels.graph_matern_kernel import GraphMaternKernel
from bax.utils.acquisition_functions import EigBax, RandomBax, VarBax
from bax.utils.gp_utils import true_func, sample
from bax.utils.dijkstra import Dijkstra
from bax.utils.visualization_utils import paint
from bax.utils.compare_module import compare


class procedure:
    """        
        Attributes:
            is_gp_on_graph: False if GP is defined on R^n, True if
              it is defined on nodes of the graph. 
            budget: Number of requests to black-box function.
            num_samples: A number of samples from GP for Monte Carlo.
            start: The index of the start vertex in the shortest path.
            finish: The index of the last vertex in the shortest path.
            model: An instance of PathwiseGPR.
            graph: An instance of GraphProcessing/MakeGrid2d class 
              from bax.utils.graph_initialization.
            alg: An instance of Dijkstra class.
            gp_params: A dictionary of parameters of the gaussian process:
              'kernel' is a kernel from gpflow.kernels 
              (if is_gp_on_graph is False), 
              'noise' is value of the noise in observations.
              'nu'/'kappa'/'sigma_f' are parameters of GraphMaternKernel
              (if is_gp_on_graph is True).
            pos, ver, edges: Positions of the vertices, 
              vertices and edges of the graph respectively.
            weights: A dictionary that contains weights of the edges.
            latent_weights: A dictionary that contains transformed weights 
              of the edges that are more preferable for gaussian processes.
            already_known: A set of edges that have already been processed
              during the execution.
            true_path: The true shortest path between start and finish
              vertices.
            true_val: Costs of the edges along the shortest path.
            elements: Either edges of the graph or nodes of the line graph
              depending on what GP is defined.
    """
    
    def __init__(self, budget, num_samples, start, finish, 
                 graph, gp_params, init=1, is_gp_on_graph=True):
        self.is_gp_on_graph = is_gp_on_graph
        self.budget = budget
        self.num_samples = num_samples
        self.start = start
        self.finish = finish
        self.pos = graph.get_pos
        self.ver =  graph.get_ver
        self.edges = graph.get_edges
        self.weights = graph.weights
        self.latent_weights = {key: true_func(value) for key, value in self.weights.items()}
        self.alg = Dijkstra(self.ver)
        self.true_path, self.true_val = self.alg.run_alg(start, finish, self.weights)[1:]
        self.graph = graph
        self.gp_params = gp_params
        if is_gp_on_graph:
            # Get line_graph to define GP on graphs
            line_graph = nx.line_graph(graph.graph)
            line_nodes = [e[0] for e in graph.get_edges]     
            self.elements = np.array(line_nodes, dtype=np.float64).reshape((-1, 2))

            # Define graph matern kernel        
            point_kernel = gpflow.kernels.Matern52()
            laplacian = sparse.csr_matrix(nx.laplacian_matrix(line_graph), 
                                          dtype=np.float64)
            eigenvalues, eigenvectors = tf.linalg.eigh(laplacian.toarray())  
            self.kernel = GraphMaternKernel((eigenvectors, eigenvalues), 
                                       nu=gp_params['nu'], 
                                       kappa=gp_params['kappa'], 
                                       sigma_f=gp_params['sigma_f'], 
                                       vertex_dim=2, 
                                       point_kernel=point_kernel)                    
        else:
            self.elements = self.edges
            if gp_params.get('kernel') is not None:         
                self.kernel = gp_params['kernel'] 
            else: 
                self.kernel = gpflow.kernels.Matern52()

        # Randomly choose first init elements (edges of graph/nodes in line graph)
        ind_of_first_elements = np.random.choice(range(len(self.elements)), init)
        first_elements = [self.elements[i] for i in ind_of_first_elements]

        # Indices of elements which cost we've already have
        self.already_known = set(ind_of_first_elements)

        # Initial dataset
        if is_gp_on_graph:
            self.data = (tf.constant([e for e in first_elements]), 
                         tf.constant([[self.latent_weights[tuple(e)]] for e in first_elements]))
        else:
            self.data = (tf.constant([e[1] for e in first_elements]), 
                         tf.constant([[self.latent_weights[e[0]]] for e in first_elements]))

        # Model
        self.model = PathwiseGPR(data=self.data, kernel=self.kernel, 
                                 noise_variance=gp_params['noise'])
            
        
    def run(self, method='EIG', num=100, is_visualize=None, 
            is_one_image=None, is_compare=None, file_path=''): 
        """
            Args:
                method: The type of acquisition function. Can be
                  'EIG'/'Random'/'Var'.
                num: Number of randomly chosen edges that will be
                  taken into account at each iteration.
                is_visualize: True for visualization of grid-shaped graph.
                is_one_image: If True, then only the image from the last
                  iteration will be saved. Otherwise, there will be
                  self.budget images.
                is_compare: If True, then the first, second and third quantiles 
                  will be saved in a file on each iteration.
                file_path: Path to the folder where the images/statistics 
                  will be saved.
        """        
                
        # Parameters that are used for visualization/comparison
        params = {
            'method': method,
            'is_gp_on_graph': self.is_gp_on_graph,
            'one_image': is_one_image,
            'start_ind': self.start,
            'finish_ind': self.finish,
            'start_ver': self.ver[self.start],
            'finish_ver': self.ver[self.finish],            
            'true_path': self.true_path,  
            'graph': self.graph,
            'alg': self.alg,
            'num_samples': self.num_samples, 
            'weights': self.weights,  
            'path': file_path,
            'elements': self.elements,
            'pos': self.pos,
            'step': 0,
            'model': self.model,
            'data': self.data
        }        
        for step in range(self.budget):
            # Randomnly choosing elements among which to select the best variant
            sub_ed_first_edition = np.random.choice(range(len(self.elements)), 
                                                    num, replace=False)
            sub_ed = []
            for i in sub_ed_first_edition:
                if i not in self.already_known:
                    sub_ed.append(i)
            if self.is_gp_on_graph:
                X = np.array([self.elements[i] for i in sub_ed])
            else:
                X = np.array([self.elements[i][1] for i in sub_ed])
            # Choosing acquisition function
            if method == 'EIG':
                acq_func = EigBax(X, self.data, self.model, self.alg, self.num_samples, 
                                  self.elements, self.gp_params['noise'], self.start, 
                                  self.finish, is_gp_on_graph=self.is_gp_on_graph)                
            if method == 'Random':
                acq_func = RandomBax(X)
            if method == 'Var':
                acq_func = VarBax(X, self.model)
            ind = np.argmax(acq_func.calculate()) 
            new_place_for_request = self.elements[sub_ed[ind]]
            if self.is_gp_on_graph:
                new_val = self.latent_weights[tuple(new_place_for_request)]
                self.already_known.add(sub_ed[ind])
                self.data = (tf.concat((self.data[0], [new_place_for_request]), 0), 
                             tf.concat((self.data[1], [[new_val]]), 0))
            else:
                new_val = self.latent_weights[new_place_for_request[0]]
                self.already_known.add(sub_ed[ind])
                self.data = (tf.concat((self.data[0], [new_place_for_request[1]]), 0), 
                             tf.concat((self.data[1], [[new_val]]), 0))
            self.model = PathwiseGPR(data=self.data, 
                                     kernel=self.kernel, 
                                     noise_variance=self.gp_params['noise'])
            # Comparison of acquisition functions            
            if is_compare:
#                 if step % 10 == 0:
#                     print(f'Finished iter i = {step}')
                params['model'] = self.model
                compare(params)
            # Make and save image/images
            if is_visualize:
#                 if step % 10 == 0:
#                     print(f'Finished iter i = {step}')
                params['step'] = step
                params['model'] = self.model
                params['data'] = self.data
                paint(params)             
                