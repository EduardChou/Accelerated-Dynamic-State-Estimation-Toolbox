import cupy as cp
import numpy as np

def nearPD_python(mat, max_iter=100, tol=1e-6):
    # Copy the input matrix to the GPU
    sym_part = cp.asarray(0.5 * (mat + mat.T))
    # Initialize the output variables
    converged = False
    iterations = 0
    normF = 0

    # Perform the iterative algorithm
    for i in range(max_iter):
        # Compute the eigenvalue decomposition of the symmetric part
        eigvals, eigvecs = cp.linalg.eigh(sym_part)

        # Compute the diagonal matrix of eigenvalues
        D = cp.diag(eigvals)

        # Compute the nearest positive definite matrix
        mat_new = eigvecs @ cp.maximum(D, cp.zeros_like(D)) @ eigvecs.T

        # Compute the Frobenius norm of the difference between the old and new matrices
        normF = cp.linalg.norm(cp.asarray(mat) - cp.asarray(mat_new), 'fro')

        # Check if the algorithm has converged
        if normF < tol:
            converged = True
            break

        # Update the symmetric part with the new matrix
        sym_part = cp.asarray(0.5 * (mat_new + mat_new.T))

        # Update the iteration counter
        iterations = i + 1

    # Copy the output matrices back to the CPU
    mat_new = cp.asnumpy(mat_new)
    normF = cp.asnumpy(normF)

    # Return the output variables
    return mat_new, normF, iterations, converged
