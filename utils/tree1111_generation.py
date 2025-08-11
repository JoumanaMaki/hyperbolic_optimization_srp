import numpy as np
import scipy.sparse as sp
import os
import pandas as pd


def generate_tree1111(gamma: float) -> None:
    feats_tree1111 = np.zeros((1111, 1000))
    labels_tree1111 = np.zeros(1111)

    # Root
    feats_tree1111[0] = np.random.randn(1000)

    # Level 1
    for i in range(1, 11):
        feats_tree1111[i] = gamma * feats_tree1111[0] + np.random.randn(1000)

    # Level 2
    for i in range(1, 11):
        for j in range(i * 10 + 1, i * 10 + 11):
            feats_tree1111[j] = gamma * feats_tree1111[i] + np.random.randn(1000)

    # Level 3
    for i in range(11, 111):
        for j in range(111 + (i - 11) * 10, 111 + (i - 11) * 10 + 10):
            feats_tree1111[j] = gamma * feats_tree1111[i] + np.random.randn(1000)

    # Edges
    parent_list = [(i - 1) // 10 for i in range(1, 1111)]
    child_list = list(range(1, 1111))
    edges_tree1111 = pd.DataFrame({'parent': parent_list, 'child': child_list})

    os.makedirs('../datasets/tree1111', exist_ok=True)
    subdir = str(gamma).replace('.', '')
    os.makedirs(f'../datasets/tree1111/g{subdir}_lp', exist_ok=True)

    edges_tree1111.to_csv(f'../datasets/tree1111/g{subdir}_lp/g{subdir}_lp.edges.csv', index=False)
    np.save(f'../datasets/tree1111/g{subdir}_lp/g{subdir}_lp.labels.npy', labels_tree1111)
    sp.save_npz(f'../datasets/tree1111/g{subdir}_lp/g{subdir}_lp.feats.npz', sp.csr_matrix(feats_tree1111))


if __name__ == "__main__":
    gammas = [0.0, 0.05, 0.10, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0]
    for gamma in gammas:
        print(f'Generating tree1111 for gamma={gamma}')
        generate_tree1111(gamma)
        print(f'Finished generating tree1111 for gamma={gamma}\n')
