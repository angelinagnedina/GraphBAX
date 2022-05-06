#!/usr/bin/env python
# coding: utf-8


import numpy as np
import tensorflow as tf


def inv_soft(x):
    return np.log(np.exp(x) - 1)


def softplus(x):
    return np.log(np.exp(x) + 1)


def true_func(x, func=lambda x: x):
    return inv_soft(func(x))


def entropy(model, X, compute_cov=True, noise=0):
    """
        Computes the entropy of the multivalriate normal.
        
        Args:
            model: Either an instance of PathwiseGPR, or, if 
              called from entropy_of_post_pred_distr, a matrix.
            X: Positions of edges/nodes of line graph
              where to compute the entropy.
            compute_cov: True if model if an instance of PathwiseGPR,
              else False.
            noise: Noise in observations.
    """
    num_ = X.shape[0]

    if compute_cov:
        # if there's a model
        det = model.predict_f(X)[1].numpy().reshape(num_) + noise
    else:
        # if entropy was called from entropy_of_post_pred_distr
        det = [model[i][i] + noise for i in range(num_)]

    for i in range(num_):
        assert det[i] > 0
        
    H = 1/2 + 1/2*np.log(2*np.pi) + [np.log(det[i])/2 for i in range(num_)]
    return H
  
    
def sample(model, alg, num_samples, elements, 
           start, finish, is_gp_on_graph=True):
    """        
        Args:
            model: An instance of PathwiseGPR.
            alg: An instance of Dijkstra class.
            num_samples: A number of samples from GP.
            elements: The edges of the graph or the nodes of the line
              graph depending on argument is_gp_on_graph.
            start: The index of the start vertex in the shortest path.
            finish: The index of the last vertex in the shortest path.
            is_gp_on_graph: False if GP is defined on R^n, True if
              it is defined on nodes of the graph.             
        Returns:
            Pairs of vertex indices along the shortest path, coordinates 
            of edges in this path and the costs of the edges along 
            these paths.
    """
    if not is_gp_on_graph:  
        paths = model.generate_paths(num_samples=num_samples, num_bases=1024)
        model.set_paths(paths)
        pos_of_edges = [e[1] for e in elements]
        samples = model.predict_f_samples(tf.constant(pos_of_edges))
        samples = samples.numpy().reshape((num_samples, -1))
    else:
        mean, cov = model.predict_f(elements, full_cov=True)
        mean = mean.numpy().reshape(-1)
        cov = cov.numpy()[0]
        samples = np.random.multivariate_normal(mean=mean, cov=cov, 
                                                size=num_samples)
    ver_pathes = dict([(i, 0) for i in range(num_samples)])
    sub_pathes = dict([(i, 0) for i in range(num_samples)])
    values = dict([(i, 0) for i in range(num_samples)])

    for i in range(num_samples):
        if not is_gp_on_graph:
            func = dict([(elements[j][0], softplus(samples[i][j])) 
                         for j in range(len(elements))])
        else:
            func = dict([(tuple(elements[j]), softplus(samples[i][j])) 
                         for j in range(len(elements))])
        ver_path_, path_, val_ = alg.run_alg(start, finish, func)
        # Reverse the vertices in the path so that 
        # they begin with start and end in finish vertices.
        ver_pathes[i] = np.array([(ver_path_[i+1], ver_path_[i]) 
                                  for i in range(len(ver_path_) - 2, -1, -1)])
        sub_pathes[i] = np.array(path_)[::-1]
        values[i] = np.array(val_)[::-1]

    return ver_pathes, sub_pathes, values


def entropy_of_pred_dist(X_new, ker, data, subseq, noise):
    data = (data[0].numpy(), data[1].numpy())
    arr = np.concatenate((data[0], subseq))

    k_T = ker(X_new, arr).numpy()
    k_ = k_T.T

    I = np.zeros((len(arr), len(arr)))
    for i in range(len(data[0])):
        I[i][i] = noise

    K = ker(arr, arr).numpy()
    K_inv = np.linalg.inv(K + I)

    cov = ker(X_new, X_new).numpy() - k_T @ K_inv @ k_ + (noise)**2*np.eye(X_new.shape[0])

    return entropy(cov, X_new, compute_cov=False, noise=noise)