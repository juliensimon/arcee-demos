"""
This module demonstrates Singular Value Decomposition (SVD) and low-rank matrix approximation.
It creates a rank-deficient matrix through matrix multiplication and shows how SVD can be used
to decompose and reconstruct the matrix with a specified number of singular values.
"""

import numpy as np

N = 1024  # Matrix dimension
top_k = 1024  # Rank of the matrix

C = np.random.rand(N, N)
print(f"Original matrix C shape: {C.shape}, Number of parameters: {C.size:,}")

# Perform SVD decomposition
U, Sigma, Vt = np.linalg.svd(C, full_matrices=True)
print("\nSVD Decomposition shapes:")
print(f"U: {U.shape}")
print(f"Sigma: {Sigma.shape}")
print(f"Vt: {Vt.shape}")

# Create diagonal matrix from Sigma and keep only top k singular values
Sigma = np.diag(Sigma)  # Convert to diagonal matrix
Sigma_k = Sigma[:top_k, :top_k]  # Keep only top k x k submatrix

# Keep only top k singular values/vectors
U_k = U[:, :top_k]
Vt_k = Vt[:top_k, :]

print("\nReduced matrices shapes:")
print(f"U_k: {U_k.shape}")
print(f"Sigma_k: {Sigma_k.shape}")
print(f"Vt_k: {Vt_k.shape}")
print(f"Total parameters in reduced form: {U_k.size + Sigma_k.size + Vt_k.size:,}")

# Reconstruct the matrix and compute error
C_reconstructed = U_k @ Sigma_k @ Vt_k
reconstruction_error = np.linalg.norm(C - C_reconstructed, "fro")
print(f"Frobenius norm: {reconstruction_error:.2e}")
