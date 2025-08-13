import numpy as np
import scipy.sparse as sp
import os
import pandas as pd


def generate_tree1111(gamma: float) -> None:
  
    """
    Generate a 10-ary tree dataset with 1,111 nodes and write edges, labels, and features to disk.

    The dataset is a 4-level rooted tree with nodes indexed in breadth-first order:
    level 0 (root) has index 0; level 1 has indices 1..10; level 2 has 11..110; and level 3 has
    111..1110, for a total of 1,111 nodes. Each node has a 1,000-dimensional feature vector.

    Feature generation follows a parent-to-child Gaussian model:
    for each edge (parent -> child),
        x_child = gamma * x_parent + ε,
    where ε ~ 𝒩(0, I_1000). The root feature x_root ~ 𝒩(0, I_1000).

    Larger values of ``gamma`` induce stronger feature correlation along the tree:
    - gamma = 0.0  → children are pure noise, independent of the parent.
    - gamma = 1.0  → children closely resemble the parent plus noise.
    Intermediate values interpolate the correlation strength.

    Files written
    -------------
    Under ``../datasets/tree1111/g{subdir}_lp`` where ``subdir = str(gamma).replace('.', '')``:

    - ``g{subdir}_lp.edges.csv`` :
        CSV with two integer columns, ``parent`` and ``child``, listing directed edges
        from each parent node to its child node. There are 1,110 edges (every node except
        the root has a single parent).

    - ``g{subdir}_lp.labels.npy`` :
        NumPy array of shape (1111,), dtype float64, containing zeros (placeholder labels).

    - ``g{subdir}_lp.feats.npz`` :
        SciPy CSR sparse matrix of shape (1111, 1000) containing the node features.
        Note that features are generated as dense Gaussian vectors and then wrapped into CSR.

    Parameters
    ----------
    gamma : float
        Strength of parent-to-child feature correlation. Must be a real number; typical
        range is [0.0, 1.0]. Higher values yield stronger similarity between a node and its
        parent. See Notes for the exact generative model.

    Returns
    -------
    None
        This function has no return value; it writes the dataset files to disk.

    Notes
    -----
    Tree structure
        The tree is 10-ary (each non-leaf has exactly 10 children) and has 4 levels:
        1 (root) + 10 (level 1) + 100 (level 2) + 1000 (level 3) = 1111 nodes total.
        Node indices are assigned level by level. The parent of node ``i >= 1`` is
        ``(i - 1) // 10`` under this indexing.

    Generative model
        Let ``x_v ∈ R^{1000}`` denote the feature of node v. The root has
        ``x_root ~ 𝒩(0, I)``. For each edge (u → v),
        ``x_v = gamma * x_u + ε_v``, where ``ε_v ~ 𝒩(0, I)`` independently across nodes.

    Reproducibility
        This function uses NumPy's global RNG without setting a seed. For reproducible
        datasets, set ``np.random.seed(seed)`` before calling, or refactor to use
        ``np.random.default_rng(seed)``.

    Examples
    --------
    >>> generate_tree1111(0.2)  # doctest: +SKIP
    Generates the dataset in ../datasets/tree1111/g02_lp/ with moderate parent-child correlation.

    See Also
    --------
    numpy.random.randn : Draw samples from the standard normal distribution.
    scipy.sparse.csr_matrix : Sparse matrix container used for saving features.
    pandas.DataFrame.to_csv : Used to write the edge list.

    References
    ----------
    .. [1] "Shedding Light on Problems in Hyperbolic Graph Learning", https://arxiv.org/pdf/2411.06688.
    .. [2] Original code repository for the paper, https://github.com/isaykatsman/Shedding-Light-Hyperbolic-Graph-Learning/tree/main.
    """

    feats_tree1111 = np.zeros((1111, 1000))     # feats_tree1111: (1111 x 1000) feature matrixs
    labels_tree1111 = np.zeros(1111)  # labels_tree1111: (1111,) label vector (all zeros here)

    # Root
    feats_tree1111[0] = np.random.randn(1000)  # Level 0 (root node) / Features ~ N(0, I)

    # Level 1 (10 children of root) -  Each child: gamma * parent_features + Gaussian noise
    for i in range(1, 11):
        feats_tree1111[i] = gamma * feats_tree1111[0] + np.random.randn(1000)  

    # Level 2 (each Level 1 node has 10 children) - Node indices for Level 2: 11..110
    for i in range(1, 11):
        for j in range(i * 10 + 1, i * 10 + 11):
            feats_tree1111[j] = gamma * feats_tree1111[i] + np.random.randn(1000)

    # Level 3 (each Level 2 node has 10 children) - Node indices for Level 3: 111..1110
    for i in range(11, 111):
        for j in range(111 + (i - 11) * 10, 111 + (i - 11) * 10 + 10):
            feats_tree1111[j] = gamma * feats_tree1111[i] + np.random.randn(1000)

    # Build edge list
    # parent(i) = (i - 1) // 10 for i >= 1
    # child_list: all non-root nodes
    parent_list = [(i - 1) // 10 for i in range(1, 1111)]
    child_list = list(range(1, 1111))
    edges_tree1111 = pd.DataFrame({'parent': parent_list, 'child': child_list})

    # Prepare output directories
    # subdir: gamma value with '.' removed (e.g., 0.05 -> "005")
    os.makedirs('../datasets/tree1111', exist_ok=True)
    subdir = str(gamma).replace('.', '')
    os.makedirs(f'../datasets/tree1111/g{subdir}_lp', exist_ok=True)

    edges_tree1111.to_csv(f'../datasets/tree1111/g{subdir}_lp/g{subdir}_lp.edges.csv', index=False) # graph structure (edges)
    np.save(f'../datasets/tree1111/g{subdir}_lp/g{subdir}_lp.labels.npy', labels_tree1111) #labels per node
    sp.save_npz(f'../datasets/tree1111/g{subdir}_lp/g{subdir}_lp.feats.npz', sp.csr_matrix(feats_tree1111)) # node features


if __name__ == "__main__":
    gammas = [0.0, 0.05, 0.10, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0]
    for gamma in gammas:
        print(f'Generating tree1111 for gamma={gamma}')
        generate_tree1111(gamma)
        print(f'Finished generating tree1111 for gamma={gamma}\n')
