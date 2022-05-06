# Realization of InfoBAX algorithm
<p float="left">
    <img src="images/image_Bax_120.png" width="35%"/>
    <img src="images/comparison_with_other_strategies_log_scale.jpg" width="30%" />
    <img src="images/comparison_with_other_strategies.jpg" width="30%" />     
</p>

This repo provides a realization of InfoBAX method proposed in the paper:

>[Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information](https://arxiv.org/abs/2104.09460)
>
>Willie Neiswanger, Ke Alexander Wang, Stefano Ermon
>
>International Conference on Machine Learning (ICML), 2021
>
>arXiv:2104.09460

There are many real world problems that can be viewed as estimating an unknown property of some function (that is called black-box function). For example, estimating the shortest path between different locations or finding the global maximum of the function. As a result of tackling this problem, the InfoBAX procedure occured.

In a few words, InfoBAX is the procedure that sequentially finds the most effective requests to the black-box function in order to evaluate some of its computable property. In this work we are interested in estimating the shortest path in graphs, where the edges have costs given by some function. For modeling such function we define gaussian processes on $\mathbb{R}^n$ or on graph nodes depending on the case. 

## Installation
To install all the dependencies, `cd` into this repo and run:

`$ pip install -r requirements.txt`

## Usage
The example of how to use infoBAX procedure can be found in examples folder in **InfoBAX_on_grid-shaped_graph** notebook. There is an example for grid-shaped graph, but the procedure can be implemented on various graphs, for example, on road graph from PEMS dataset. 

<img src="images/Estimated_shortest_path.jpg" width ="50%">

The way to get a visualization for a grid-shaped graph can also be found in that notebook, you can either get a one image, or a few to make a gif such as the following:

<img src="images/visualization_of_infoBAX.gif"/>

There you may also find a procedure to compare different strategies for finding new positions to request to black-box function. An example of such comparison is given in two plots at the beginning of the README file.
