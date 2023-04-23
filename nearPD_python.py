import numpy as np

def nearPD_python(mat, max_iter=100, tol=1e-6):
    # Get the symmetric part of the matrix
    sym_part = 0.5 * (mat + mat.T)

    # Initialize the output variables
    converged = False
    iterations = 0
    normF = 0

    # Perform the iterative algorithm
    for i in range(max_iter):
        # Compute the eigenvalue decomposition of the symmetric part
        eigvals, eigvecs = np.linalg.eigh(sym_part)

        # Compute the diagonal matrix of eigenvalues
        D = np.diag(eigvals)

        # Compute the nearest positive definite matrix
        mat_new = eigvecs @ np.maximum(D, np.zeros_like(D)) @ eigvecs.T

        # Compute the Frobenius norm of the difference between the old and new matrices
        normF = np.linalg.norm(mat - mat_new, 'fro')

        # Check if the algorithm has converged
        if normF < tol:
            converged = True
            break

        # Update the symmetric part with the new matrix
        sym_part = 0.5 * (mat_new + mat_new.T)

        # Update the iteration counter
        iterations = i + 1

    # Return the output variables
    return mat_new, normF, iterations, converged