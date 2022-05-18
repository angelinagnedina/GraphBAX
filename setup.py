from setuptools import setup, find_packages

requirements = (
    'numpy>=1.20.3',
    'tensorflow>=2.8.0',
    'gpflow>=2.4.0',
    'scipy>=1.7.1',
    'tensorflow-probability>=0.16.0',
    'networkx>=2.6.3',
    'matplotlib>=3.4.3',
    'sortedcontainers>=2.4.0',
    'ipywidgets>=7.6.5',
    'gpflow_sampling @ git+https://github.com/j-wilson/GPflowSampling.git',
    'graph_matern @ git+https://github.com/vabor112/Graph-Gaussian-Processes.git'
)

setup(name='bax',
      version='1.0',
      packages=find_packages(),
      python_requires='>=3.9.7',
      install_requires=requirements)
