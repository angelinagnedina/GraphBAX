#!/usr/bin/env python
# coding: utf-8


import numpy as np
from bax.utils.gp_utils import entropy, sample, entropy_of_pred_dist


def EigBax(X, data, model, alg, num_sam, elements, 
           noise, start, finish, is_gp_on_graph=True):
    """
        The aqcuisition function from the paper.
        
        Args:
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
        Returns:
            The result of the aqcuisition function on X.
    """
    
    first = entropy(model, X, noise=noise)
    sub_s= sample(model, alg, num_sam, elements, 
                  start, finish, is_gp_on_graph)[:2]
    if is_gp_on_graph:
        diff_entr = [entropy_of_pred_dist(X, model.kernel, data, 
                                          sub_s[0][i], noise) for i in range(num_sam)]
    else:
        diff_entr = [entropy_of_pred_dist(X, model.kernel, 
                                          data, sub_s[1][i], noise) for i in range(num_sam)]
    second = np.sum(diff_entr, axis=0)/num_sam
    return first - second


def RandomBax(X):
    """
        The aqcuisition function taken as baseline.
        
        Args:
            X: Candidates for the next query to black-box function.
        Returns:
            The result of the aqcuisition function on X.
        
    """
    ind = np.random.choice(range(len(X)), 1)
    result = [0 if i != ind else 1 for i in range(len(X))]
    return result


def VarBax(X, model):
    """
        The aqcuisition function taken as baseline.
        
        Args:
            X: Candidates for the next query to black-box function.
        Returns:
            The result of the aqcuisition function on X.
    """
    
    var = model.predict_f(X)[1].numpy()
    return var
