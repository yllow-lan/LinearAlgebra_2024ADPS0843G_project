import numpy as np

def page_rank(M, d=0.85, tol=1.0e-6):
    """
    M: Transition Matrix (n x n)
    d: Damping factor (teleportation probability)
    tol: Tolerance for convergence
    """
    n = M.shape[0]
    
    # Initial Rank Guess: Equal probability 1/n
    v = np.ones(n) / n
    
    # The Google Matrix G
    # G = d*M + (1-d)*E, where E has 1/n everywhere
    E = np.ones((n, n)) / n
    G = d * M + (1 - d) * E
    
    iteration = 0
    while True:
        iteration += 1
        v_new = np.dot(G, v)
        
        # Check for convergence (if ranks stop changing)
        if np.linalg.norm(v_new - v) < tol:
            break
            
        v = v_new
        
    return v, iteration

# --- Test with 3 Pages ---
# A links to B
# B links to C
# C links to A (Circular loop)
M = np.array([
    [0, 0, 1],  # Col 1: Links from A (A->B only, so 0 to A, 1 to B, 0 to C? 
                # No, M_ij is Prob(j -> i). 
    [1, 0, 0],  # Row 1 is In-links to A. Row 2 is In-links to B.
    [0, 1, 0]   # Row 3 is In-links to C.
])

# Note: In standard algebra M is often Column Stochastic.
# Here: Col 1 (From A) -> Goes to B (Row 2). So M[1,0] = 1. Correct.

ranks, iters = page_rank(M)
print(f"Final Ranks: {ranks}")
print(f"Converged in {iters} iterations.")