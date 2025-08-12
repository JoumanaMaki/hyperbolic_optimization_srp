# Hyperbolic Optimization

This repository contains the implementation and experiments for our Student Research Project (SRP) on **Optimization in Hyperbolic Neural Networks** at the University of Hildesheim (2025).

We compare three optimization strategies in hyperbolic space:

- **Riemannian Optimization**
- **Euclidean Parametrization**
- **Curvature-Aware Optimization**

These techniques are tested across graph datasets (e.g., Cora, Tree1111) and image datasets (e.g., MNIST, CIFAR-100) using fully hyperbolic neural architectures.

# Setup the environment

## 1. Install UV and create virtual environment

1. Install **uv** package (not inside any other venv!)
   
    ```pipx install uv```

   **Note:** check [docs](https://docs.astral.sh/uv/getting-started/installation/) if you encounter problems


3. Create the environment 

    ```uv venv```


4. Activate the environment


    * Windows: ```source .venv/Scripts/activate```
    * Linux-based: ```source .venv/bin/activate```


5. Install the dependencies from lock file:

    ```uv sync --locked```


6. *(Optional)* Sometimes **pip** module is missing, in that case:

    1. install it 

        ```uv pip install pip```

    2. Or just use 

        ```uv pip list```

## 2. Add new packages

1. ```uv add <package_name>```

2. Commit the ```pyproject``` and ```lock``` files and push them

## 3. Pre-commit hook

Install pre-commit to make the code style consistent among developers.

Run ```pre-commit install``` once you set up the venv

This will automatically format your code to a beautiful standardized format when you try to commit.

# 📦 Tools & Libraries
- [PyTorch](https://pytorch.org/)
- [Geoopt](https://geoopt.readthedocs.io/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (for graph models)


# 👥 Authors
Elvin Guseinov, Joumana Makki, Lavrentii Grigorian, Mustafa Ahmed, Omar Adardour  
Supervisor: Ahmad Bdeir

