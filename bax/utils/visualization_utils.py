#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection as LC
from bax.utils.gp_utils import sample
from bax.utils.graph_initialization import rosenbrock


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
    cs = ax.contourf(x, y, rosenbrock((x, y)), 
                     colors = ['#F5F1F8', '#F1E1FC', '#F0C5EE', '#E4BBE2', 
                               '#C9A5C8', '#A386A2', '#786277'])
    cbar = fig.colorbar(cs, ax = ax)

    
def plot_graph(ax, pos, edges, start, goal):
    # plot edges
    color = (0.75, 0.75, 0.75, 0.1)
    lc = LC(edges, colors=[color]*len(edges), linewidths=1.0)
    ax.add_collection(lc)

    # plot vertices
    ax.scatter(*pos.T, color=(0, 0, 0, 1),
               marker='.', facecolors='none', s=20)

    # plot start and goal vertices
    ax.scatter(*start.position, color='#FF530A', 
               label="Start", s=150)
    ax.scatter(*goal.position, color='#21FF65', 
               label='Goal', s=150)

    ax.grid(False)
    return


def plot_path(
    ax,
    path,
    path_color=(0, 0, 0, 1.),
    linewidths=2,
    linestyle='dotted',
    plot_vertices=False,
    label=None,
):
    # plot path taken
    path_lines = edges_of_path(path)
    path_lc = LC(
        path_lines,
        colors=[path_color]*len(path_lines),
        linewidths=linewidths,
        linestyle=linestyle,
        label=label,
    )
    ax.add_collection(path_lc)

    # plot visited vertices
    if plot_vertices:
        ax.scatter(*positions_of_path(path).T, color=(0, 0, 0, 1))
    return


def paint(params):
    """
        Method for visualizing the work of the procedure. It 
        saves images to the folder corresponding to file_path.
    """
    edges_ = [(e[0][1], e[1][1]) for e in params['graph'].edges]
    start, finish = params['start_ver'], params['finish_ver']
    sampled_paths= sample(params['model'], params['alg'], 15, 
                          params['elements'], params['start_ind'], 
                          params['finish_ind'], False)[1]
    sampled_paths = [sampled_paths[j] for j in range(15)]

    fig, ax = plt.subplots(figsize=(8.9, 7))
    plot_contourf(fig, ax, (-2, 2), (-1, 4))

    plot_graph(ax, params['pos'], edges_, start, finish)

    plot_path(
        ax,
        params['true_path'],
        path_color=(0.2, 0.2, 0.2, 1),
        linewidths=2,
        linestyle='--',
        label='True shortest path',
    )


    data_ = (params['data'][0].numpy(), params['data'][1].numpy())

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
        plot_path(ax, path, path_color=(0, 0, 1, weight), 
                  linewidths=2, linestyle="-")

    ax.set(ylim=[-1.2, 4.2], xlim=[-2.2, 2.2]) 

#     Plot title
    ax.set_title("InfoBAX with Dijkstra's Algorithm")

    # Turn off ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if params['one_image']:
        fig.savefig(params['path'] + f'image_Bax.png', bbox_inches='tight')
    else:
        step = params['step']
        fig.savefig(params['path'] + f'image_Bax_{step}.png', bbox_inches='tight')
    plt.close()