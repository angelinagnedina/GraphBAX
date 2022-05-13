# Reproducing InfoBAX for Graph Shortest Paths
<p float="left">
<!--     <img src="images/image_Bax_120.png" width="35%"/> -->
    <img src="images/visualization_of_infoBAX.gif"/>
    <img src="images/comparison_log_scale" width="55%" height="350"/>
<!--     <img src="images/comparison_with_other_strategies_log_scale.jpg" /> -->
<!--     <img src="images/comparison_with_other_strategies.jpg" width="30%" />      -->
</p>

This repo provides an implementation of the InfoBAX ([Neiswenger et al.](https://arxiv.org/abs/2104.09460)) featuring some additional experiments and benchmarks.

There are many real world problems that can be viewed as estimating an unknown property of a black-box function. For example, estimating the shortest path between different locations or finding the global maximum of the function. As a result of tackling this problem, the InfoBAX procedure occured.

In a few words, InfoBAX is the procedure that sequentially finds the most effective requests to the black-box function in order to evaluate some of its computable properties. In this work we are interested in estimating the shortest path in graphs, where the edges have costs given by an unknown function. For modeling such a function we define Gaussian processes on <img src="https://latex.codecogs.com/gif.latex?\mathbb{R}^n" /> or directly on the edges of the graph.

The original paper is

>Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information
>
>Willie Neiswanger, Ke Alexander Wang, Stefano Ermon
>
>International Conference on Machine Learning (ICML), 2021
>
>[arXiv:2104.09460](https://arxiv.org/abs/2104.09460)

## Installation
To install the package along with all of its dependencies

`pip instal git+https://github.com/angelinagnedina/BAX_realization.git`

## Usage
<<<<<<< HEAD
Below is an outline of how to use InfoBAX on your graph:

```
from bax.utils import graph_initialization
from bax.infoBax import procedure

gp_params = {
    'noise': noise, # float
    'kernel': kernel # gpflow kernel
}
# G is a networkx Graph with weighted edges
graph = graph_initialization.GraphProcessing(G) 
graph.create_graph()

bax = procedure(budget, num_samples, start, 
                finish, graph, gp_params, 
                init=1, is_gp_on_graph=True)
bax.run(num=number_of_edges_to_check)
model = bax.model # PathwiseGPR model
```

## Examples
The example of how to use infoBAX procedure can be found in examples folder in **InfoBAX_on_grid-shaped_graph** notebook. There is an example for a grid-shaped graph, but the procedure can be implemented on various graphs, for instance, on a road graph from PEMS dataset. 
=======
The example of how to use infoBAX procedure can be found in [examples](examples) folder in [InfoBAX_on_grid-shaped_graph](examples/InfoBAX_on_grid-shaped_graph.ipynb) notebook. There is an example for grid-shaped graph, but the procedure can be implemented on various graphs, for example, on road graph from PEMS dataset. 
>>>>>>> c2c1e3ad3b306dfa4fe348f59a4f9d0f2589efbb

<img src="images/Estimated_shortest_path.jpg" width ="50%">

The way to get a visualization for a grid-shaped graph can also be found in that notebook, you can either get a one image, or a few to make a gif such as the one presented at the beginning of the README file. There you may also find a procedure to compare different strategies for finding new positions to request to black-box function. An example of such comparison is given in a plot above.
