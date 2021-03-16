import numpy as np
import jacobi


def matrix_mul(a, b):
        ar, ac = a.shape  # n_rows * n_cols
        br, bc = b.shape  # n_rows * n_cols
        assert ac == br  # rows == columns
        c = np.zeros((ar, bc))
        for i in range(ar):
                for j in range(bc):
                        for k in range(ac):  # or br
                                c[i, j] += a[i, k] * b[k, j]
        return c


def scratch_fit_transform(X,n_components):
        mean = np.mean(X, axis = 0)
        X_std = (X - mean)
        cov_mat = np.cov(X_std, rowvar = 0)
        eig_vals, eig_vecs = jacobi.Jacobi(cov_mat)
        eigen_values_sorted = np.argsort(-eig_vals)
        components = eig_vecs[eigen_values_sorted[:n_components]]
        X_std_pca = matrix_mul(X_std,components.T)
        return (X_std_pca),components


def scratch_transform(X,xcv):
        X_transformed = matrix_mul(X, xcv.T)
        return X_transformed