"""
Scalar Modeling and Environmental Simulation Engine

This module contains the core simulation functions for modeling information
persistence in networked systems with configurable coupling strengths.

The simulation uses a batch-vectorized approach for efficient computation
of mutual information dynamics across node ensembles.
"""

import numpy as np
from scipy.special import expit


# ==========================================
# PART 1: VECTORIZED SIMULATOR (The Engine)
# ==========================================

def init_equilibrium_batch(N, M, rng):
    """
    Initialize a batch of M binary state vectors of length N.
    
    Args:
        N: Number of nodes in the network
        M: Number of ensemble members (batch size)
        rng: NumPy random generator instance
        
    Returns:
        Array of shape (M, N) with random binary states
    """
    return rng.integers(0, 2, size=(M, N), dtype=np.int8)


def update_step_batch(X, c_arr, beta, theta_arr, rng):
    """
    Perform one update step for the entire batch.
    
    Uses a probabilistic update rule based on neighbor states and coupling strengths.
    
    Args:
        X: Current state array of shape (M, N)
        c_arr: Coupling strength array of length N-1 (one per edge)
        beta: Inverse temperature parameter controlling stochasticity
        theta_arr: Threshold array of length N
        rng: NumPy random generator instance
        
    Returns:
        Updated state array of shape (M, N)
    """
    M, N = X.shape
    left = np.zeros_like(X)
    right = np.zeros_like(X)
    left[:, 1:] = X[:, :-1] * c_arr
    right[:, :-1] = X[:, 1:] * c_arr
    neighbor_sum = left + right
    bias = beta * (neighbor_sum - theta_arr[None, :])
    p1 = expit(bias)
    U = rng.random(size=(M, N))
    return (U < p1).astype(np.int8)


def run_simulation_batch(N, c_arr, beta, theta_arr, M, T, source_j, master_seed=12345):
    """
    Run a batch simulation and collect joint count statistics.
    
    Args:
        N: Number of nodes
        c_arr: Coupling strength array of length N-1
        beta: Inverse temperature parameter
        theta_arr: Threshold array of length N
        M: Number of ensemble members
        T: Number of time steps
        source_j: Index of the source node
        master_seed: Random seed for reproducibility
        
    Returns:
        Count array of shape (T, N, 2, 2) containing joint counts
        for (source_bit, node_state) pairs at each timestep and node
    """
    rng = np.random.default_rng(master_seed)
    counts = np.zeros((T, N, 2, 2), dtype=np.int64)
    X = init_equilibrium_batch(N, M, rng)
    A = rng.integers(0, 2, size=M, dtype=np.int8)
    X[:, source_j] = A

    for t in range(T):
        for a in (0, 1):
            mask_a = (A == a)
            if not np.any(mask_a):
                continue
            sub = X[mask_a]
            ones = np.sum(sub, axis=0)
            zeros = sub.shape[0] - ones
            counts[t, :, a, 1] += ones
            counts[t, :, a, 0] += zeros
        X = update_step_batch(X, c_arr, beta, theta_arr, rng)
    return counts


def run_simulation_chunked(N, c_arr, beta, theta_arr, M, T, source_j,
                           master_seed=12345, chunk_size=500, continuous_source=False):
    """
    Run a chunked simulation for large ensemble sizes.
    
    Processes the simulation in chunks to manage memory usage while
    maintaining reproducibility through careful seed management.
    
    Args:
        N: Number of nodes
        c_arr: Coupling strength array of length N-1
        beta: Inverse temperature parameter
        theta_arr: Threshold array of length N
        M: Total number of ensemble members
        T: Number of time steps
        source_j: Index of the source node
        master_seed: Random seed for reproducibility
        chunk_size: Size of each processing chunk
        continuous_source: If True, continuously clamp source node
        
    Returns:
        Count array of shape (T, N, 2, 2) containing aggregated joint counts
    """
    main_rng = np.random.default_rng(master_seed)
    total_counts = np.zeros((T, N, 2, 2), dtype=np.int64)

    num_chunks = M // chunk_size + (1 if M % chunk_size != 0 else 0)

    for chunk_idx in range(num_chunks):
        current_chunk_size = min(chunk_size, M - chunk_idx * chunk_size)
        if current_chunk_size == 0:
            continue

        # Generate unique seed for this chunk from the master RNG
        chunk_seed = main_rng.integers(0, 2**31)
        chunk_rng = np.random.default_rng(chunk_seed)

        X = init_equilibrium_batch(N, current_chunk_size, chunk_rng)
        A = chunk_rng.integers(0, 2, size=current_chunk_size, dtype=np.int8)
        X[:, source_j] = A

        for t in range(T):
            # accumulate counts across ensemble
            for a in (0, 1):
                mask_a = (A == a)
                sub = X[mask_a]
                if sub.size > 0:
                    ones = np.sum(sub, axis=0)
                    total_counts[t, :, a, 1] += ones
                    total_counts[t, :, a, 0] += sub.shape[0] - ones

            X = update_step_batch(X, c_arr, beta, theta_arr, chunk_rng)
            if continuous_source:
                X[:, source_j] = A  # Continuous clamping

    return total_counts


def compute_mi_from_counts(counts, M):
    """
    Calculate Mutual Information I(A; X_i) from joint counts.
    
    Args:
        counts: Array of shape (T, N, 2, 2) containing joint counts
        M: Total number of ensemble members
        
    Returns:
        Array of shape (T, N) containing mutual information values
    """
    epsilon = 1e-12
    
    # Vectorized computation: p_ax has shape (T, N, 2, 2)
    p_ax = counts.astype(float) / M
    
    # Marginals: p_a has shape (T, N, 2), p_x has shape (T, N, 2)
    p_a = p_ax.sum(axis=3)  # Sum over x
    p_x = p_ax.sum(axis=2)  # Sum over a
    
    # Compute denominator p(a) * p(x) for all combinations
    # Need shape (T, N, 2, 2) where [t, n, a, x] = p_a[t,n,a] * p_x[t,n,x]
    denom = p_a[:, :, :, np.newaxis] * p_x[:, :, np.newaxis, :]
    
    # Compute MI only where both p and denom are > epsilon
    valid = (p_ax > epsilon) & (denom > epsilon)
    
    # Compute ratio safely - use 1 where invalid to avoid division issues
    ratio = np.divide(p_ax, denom, out=np.ones_like(p_ax), where=valid)
    
    # Compute log term safely - only compute log where valid
    log_term = np.zeros_like(p_ax)
    np.log2(ratio, out=log_term, where=valid)
    
    # MI = sum over a, x of p(a,x) * log2(p(a,x) / (p(a) * p(x)))
    I = np.sum(p_ax * log_term, axis=(2, 3))
    
    return I


# Alias for consistency with notebook naming
compute_mutual_information_from_counts = compute_mi_from_counts


def compute_tau(I_matrix, threshold=1e-2):
    """
    Compute persistence times (tau) for each node.
    
    Finds the first timestep where mutual information drops below threshold.
    
    Args:
        I_matrix: Mutual information array of shape (T, N)
        threshold: Information threshold for decay detection
        
    Returns:
        Array of length N containing persistence times
    """
    T, N = I_matrix.shape
    
    # Create boolean mask where MI is below threshold
    below_threshold = I_matrix < threshold
    
    # Find first occurrence of True in each column
    # argmax returns 0 if no True exists, so we need to handle that case
    first_decay = np.argmax(below_threshold, axis=0)
    
    # Check if decay actually occurred (argmax returns 0 both when first element
    # is True and when no element is True)
    never_decayed = ~np.any(below_threshold, axis=0)
    
    # Set tau to T for nodes that never decayed
    taus = np.where(never_decayed, T, first_decay)
    
    return taus


# Deprecated alias for consistency with notebook naming.
# Use `compute_tau` instead of `persistence_times` in new code.
persistence_times = compute_tau


def create_pocket_coupling(N, pocket_center, pocket_half_width, pocket_strength, baseline=1.0):
    """
    Create a coupling array with a localized "pocket" of stronger coupling.
    
    Args:
        N: Number of nodes
        pocket_center: Center index of the pocket
        pocket_half_width: Half-width of the pocket region
        pocket_strength: Coupling strength within the pocket
        baseline: Baseline coupling strength outside pocket
        
    Returns:
        Coupling array of length N-1
    """
    c = np.ones(N - 1) * baseline
    start = max(0, pocket_center - pocket_half_width)
    end = min(N - 1, pocket_center + pocket_half_width)
    c[start:end] = pocket_strength
    return c


if __name__ == "__main__":
    # Example usage demonstrating the simulation
    import time
    
    print("Running example simulation...")
    
    # Parameters
    N = 21
    M = 1000
    T = 60
    source_j = 10
    beta = 1.5
    theta_arr = np.ones(N) * 1.0
    pocket_strength = 10
    
    # Create pocket coupling
    c_pocket = create_pocket_coupling(N, N // 2, 3, pocket_strength)
    
    t0 = time.perf_counter()
    counts = run_simulation_chunked(
        N, c_pocket, beta, theta_arr,
        M, T, source_j,
        master_seed=20251123,
        chunk_size=500
    )
    I = compute_mi_from_counts(counts, M)
    tau = compute_tau(I, threshold=1e-3)
    elapsed = time.perf_counter() - t0
    
    # Summary statistics
    mid = N // 2
    half = 3
    inside_idx = list(range(mid - half, mid + half))
    outside_idx = [i for i in range(N) if i not in inside_idx]
    
    mean_inside = tau[inside_idx].mean()
    mean_outside = tau[outside_idx].mean()
    
    print(f"Simulation completed in {elapsed:.2f}s")
    print(f"Pocket strength: {pocket_strength}, Beta: {beta}")
    print(f"Mean tau inside pocket: {mean_inside:.4f}")
    print(f"Mean tau outside pocket: {mean_outside:.4f}")
    print(f"Contrast (inside - outside): {mean_inside - mean_outside:.4f}")
