# Hyperbolic Optimization

This repository contains the implementation and experiments for our Student Research Project (SRP) on **Optimization in Hyperbolic Neural Networks** at the University of Hildesheim (2025).

We compare three optimization strategies in hyperbolic space:

- **Riemannian Optimization**
- **Euclidean Parametrization**
- **Curvature-Aware Optimization**

These techniques are tested across graph datasets (e.g., Cora, Tree1111) and image datasets (e.g., MNIST, CIFAR-100) using fully hyperbolic neural architectures.

## Setup the environment

1. Install **uv** package (not inside any other venv!)

    ```pipx install uv```


2. Create the environment 

    ```uv venv```


3. Activate the environment


    * Windows: ```source .venv/Scripts/activate```
    * Linux-based: ```source .venv/bin/activate```


4. Install the dependencies from lock file:

    ```uv sync --locked```


5. *(Optional)* Sometimes **pip** module is missing, in that case:

    1. install it 

        ```uv pip install pip```

    2. Or just use 
    
        ```uv pip list```

## 📦 Tools & Libraries
- [PyTorch](https://pytorch.org/)
- [Geoopt](https://geoopt.readthedocs.io/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (for graph models)

## 👥 Authors
Elvin Guseinov, Joumana Makki, Lavrentii Grigorian, Mustafa Ahmed, Omar Adardour  
Supervisor: Ahmad Bdeir

