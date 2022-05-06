#!/usr/bin/env python
# coding: utf-8


import numpy as np
from bax.utils.gp_utils import entropy, sample, entropy_of_pred_dist


class EigBax:
    """
        The aqcuisition function from the paper.
        
        Attributes:
            X: Candidates for the next query to black-box function.
            data: The dataset that contains edges that have already
              been evaluated.
            model: An instance of PathwiseGPR.
            alg: An instance of Dijkstra class.
            num_sam: A number of samples from GP.
            elements: The edges of the graph or the nodes of the line
              graph depending on the argument is_gp_on_graph.
            noise: Noise in observations.
            start: The index of the start vertex in the shortest path.
            finish: The index of the last vertex in the shortest path. 
            is_gp_on_graph: False if GP is defined on R^n, True if
              it is defined on nodes of the graph.
    """
    
    def __init__(self, X, data, model, alg, num_sam, elements, 
                 noise, start, finish, is_gp_on_graph=True):
        self.X = X
        self.data = data
        self.alg = alg
        self.model = model
        self.num_sam = num_sam
        self.elements = elements
        self.noise = noise
        self.start = start
        self.finish = finish
        self.is_gp_on_graph = is_gp_on_graph
    
    def calculate(self):
        """
            Calculates the result of the aqcuisition function on X.
        """
        first = entropy(self.model, self.X, noise=self.noise)
        sub_s= sample(self.model, self.alg, self.num_sam, self.elements, 
                      self.start, self.finish, self.is_gp_on_graph)[:2]
        if self.is_gp_on_graph:
            diff_entr = [entropy_of_pred_dist(self.X, self.model.kernel, 
                                              self.data, sub_s[0][i], self.noise) for i in range(self.num_sam)]
        else:
            diff_entr = [entropy_of_pred_dist(self.X, self.model.kernel, 
                                              self.data, sub_s[1][i], self.noise) for i in range(self.num_sam)]
        second = np.sum(diff_entr, axis=0)/self.num_sam
        return first - second

    
class RandomBax:
    """
        The aqcuisition function taken as baseline.
        
        Attributes:
            X: Candidates for the next query to black-box function.
    """
    
    def __init__(self, X):
        self.X = X
        
    def calculate(self):
        """
            Calculates the result of the aqcuisition function on X.
        """
        ind = np.random.choice(range(len(self.X)), 1)
        result = [0 if i != ind else 1 for i in range(len(self.X))]
        return result

    
class VarBax:
    """
        The aqcuisition function taken as baseline.
        
        Attributes:
            X: Candidates for the next query to black-box function.
    """
    
    def __init__(self, X, model):
        self.X = X
        self.model = model
        
    def calculate(self):
        """
            Calculates the result of the aqcuisition function on X.
        """
        var = self.model.predict_f(self.X)[1].numpy()
        return var