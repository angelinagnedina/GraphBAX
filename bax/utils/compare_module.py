#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from bax.utils.gp_utils import sample


def compare(params):
    """
        Args:
            params['path']: Path to the folder where the data
              will be stored.
            params['method']: Info about the sampling strategy,
              will be used to name a file.
            params['num_samples']: Number of samples to take from
              GP posterior in order to get data.
            params['model']: GPR model.
            params['alg']: Dijkstra algorithm.
            params['elements']: Edges of the graph/nodes of the line graph.
            params['start/finish_ind']: Indices of the start/finish 
              vertices.
            params['weights']: Dictionary containing true costs of
              the edges of the graph.
            params['is_gp_on_graph']: False if GP is defined on R^n, 
              True if it is defined on nodes of the graph.
    """
    
    file_path = params['path'] + params['method'] + '.txt'
    num_samples = params['num_samples']
    ver = sample(params['model'], params['alg'], num_samples, 
                 params['elements'], params['start_ind'], 
                 params['finish_ind'], params['is_gp_on_graph'])[0]
    for key, value in ver.items():
        ver[key] = value.astype(np.int32)
    weights_in_pathes = [[] for i in range(params['num_samples'])]
    for i in range(num_samples):
        for e in ver[i]:
            if e[0] < e[1]:
                weights_in_pathes[i].append(params['weights'][(e[0], e[1])])
            else:
                weights_in_pathes[i].append(params['weights'][(e[1], e[0])])
    costs_of_pathes = [np.sum(weights_in_pathes[i], axis = 0) for i in range(num_samples)]
    costs_of_pathes.sort()
    q_1 = costs_of_pathes[int(0.25*num_samples)]
    q_2 = costs_of_pathes[int(0.5*num_samples)]
    q_3 = costs_of_pathes[int(0.75*num_samples)]
    with open(file_path, 'a') as f:
        f.write(str(q_1) + ' ' + str(q_2) + ' ' + str(q_3) + '\n') 


def get_statistics(dict_of_file_names, path_to_dir, 
                   num_of_iter, num_of_trials=1):  
    """
        Args:
            dict_of_file_names: Dictionary where keys will be 
              used in legend during visualization and values
              are names of files which stores quantiles.
            path_to_dir: Path to the directory containing the
              files mentioned above.
            num_of_iter: Number of iteration of infoBAX.
            num_of_trials: Number of experiments.
    """
    
    for method, file_name in dict_of_file_names.items():
        q_1_val = [[] for _ in range(num_of_iter)]
        med_val = [[] for _ in range(num_of_iter)]
        q_3_val = [[] for _ in range(num_of_iter)]
    
        for i in range(num_of_trials):
            cnt = 0
            with open(path_to_dir + file_name, 'r') as f:
                for line in f.readlines()[i*num_of_iter:(i+1)*num_of_iter]:
                    q_1, med, q_3 = list(map(float, line.split()))
                    q_1_val[cnt].append(q_1)
                    med_val[cnt].append(med)
                    q_3_val[cnt].append(q_3)
                    cnt += 1

        q_1_val = [sorted(item) for item in q_1_val]
        med_val = [sorted(item) for item in med_val]
        q_3_val = [sorted(item) for item in q_3_val]
        num = num_of_trials//2
        
        with open(path_to_dir + 'statistics.txt', 'a') as f:
            f.write(method + '\n')
            
        for i in range(num_of_iter):
            with open(path_to_dir + 'statistics.txt', 'a') as f:
                f.write(str(q_1_val[i][num]) + ' ' + str(med_val[i][num]) 
                        + ' ' + str(q_3_val[i][num]) + '\n')


def get_plot(path_to_dir, dict_of_func, paint_params, true_val, num_of_iter):
    """
        Args:
            path_to_dir: Path to the directory containing 
              'statistics.txt' file.
            dict_of_func: Dictionary where keys will be 
              used in legend and values are colors of the
              lines representing different strategies.
            pain_params: Info about figsize, dpi, linewidth, 
              fontsize, location of the legend, title, type of scale
              and the name of the resulting plot.
            true_val: True cost of the shortest path.
            num_of_iter: Number of iteration of infoBAX.
    """
    
    lines = []
    with open(path_to_dir + 'statistics.txt', 'r') as f:
        lines = f.readlines()
    all_values = []
    num_of_funcs = len(lines)//(num_of_iter + 1)
    
    plt.figure(figsize=paint_params['figsize'], dpi=paint_params['dpi'])
    plt.title(paint_params['title'])
    plt.plot(range(0, num_of_iter), [true_val]*num_of_iter, 
             linewidth=2*paint_params['linewidth'], color='black', 
             label='True cost')    

    for i in range(num_of_funcs):
        key = lines[i*(num_of_iter+1)].strip()
        if dict_of_func.get(key) is not None: 
            color = dict_of_func[key]
            q_1 = []
            med = []
            q_3 = []
            for line in lines[1 + i*(num_of_iter + 1):(i + 1)*(num_of_iter + 1)]:
                a, b, c = list(map(float, line.split()))
                q_1.append(a)
                med.append(b)
                q_3.append(c)
            if paint_params['scale'] == 'semilogy':
                plt.semilogy(range(0, num_of_iter), med, linewidth=paint_params['linewidth'], 
                     color=color, label=key)
            else:
                plt.plot(range(0, num_of_iter), med, linewidth=paint_params['linewidth'], 
                         color=color, label=key)
            plt.fill_between(range(0, num_of_iter), q_1, q_3, 
                             color=color, alpha=0.3)
            
    plt.legend(loc=paint_params['loc'], fontsize=paint_params['fontsize'])    
    plt.xlim(0, num_of_iter)
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.savefig(path_to_dir + paint_params['name'], 
                bbox_inches='tight', format='PNG')
    plt.show()
    