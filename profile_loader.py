"""
Observer Profile Loader for Scalar Modeling Simulation

This module provides utilities to load and execute observer profiles
defined in observer_profiles.json, enabling reproducible experiments
with standardized configurations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import scalar_simulation as sim


# Default random seed for reproducible experiments
# Using 42 as a conventional choice (popularized by "The Hitchhiker's Guide to the Galaxy")
# This ensures consistent results across runs when no specific seed is provided
DEFAULT_RANDOM_SEED = 42


def load_profiles(profile_path: str = "observer_profiles.json") -> Dict[str, Any]:
    """
    Load observer profiles from JSON file.
    
    Args:
        profile_path: Path to the observer profiles JSON file
        
    Returns:
        Dictionary containing all profile definitions
    """
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data


def get_profile(profile_name: str, profile_path: str = "observer_profiles.json") -> Dict[str, Any]:
    """
    Retrieve a specific observer profile by name.
    
    Args:
        profile_name: Name of the profile to load
        profile_path: Path to the observer profiles JSON file
        
    Returns:
        Dictionary containing the profile configuration
        
    Raises:
        ValueError: If profile name is not found
    """
    data = load_profiles(profile_path)
    
    for profile in data['profiles']:
        if profile['name'] == profile_name:
            return profile
    
    raise ValueError(f"Profile '{profile_name}' not found in {profile_path}")


def list_profiles(category: Optional[str] = None, 
                 profile_path: str = "observer_profiles.json") -> List[str]:
    """
    List available observer profile names.
    
    Args:
        category: Optional category filter (e.g., 'baseline', 'single_pocket')
        profile_path: Path to the observer profiles JSON file
        
    Returns:
        List of profile names
    """
    data = load_profiles(profile_path)
    profiles = data['profiles']
    
    # If category filter is provided, validate it exists
    if category:
        available_categories = set(p.get('category') for p in profiles)
        if category not in available_categories:
            raise ValueError(
                f"Category '{category}' not found. "
                f"Available categories: {sorted(available_categories)}"
            )
        profiles = [p for p in profiles if p.get('category') == category]
    
    return [p['name'] for p in profiles]


def create_coupling_array(coupling_config: Dict[str, Any], N: int) -> np.ndarray:
    """
    Create a coupling array from profile configuration.
    
    Args:
        coupling_config: Coupling configuration from profile
        N: Number of nodes in the network
        
    Returns:
        Coupling array of length N-1
    """
    c_type = coupling_config['type']
    baseline = coupling_config['baseline']
    c_arr = np.ones(N - 1) * baseline
    
    if c_type == 'uniform':
        # Already initialized to baseline
        pass
    
    elif c_type in ['localized_pocket', 'multi_pocket']:
        pockets = coupling_config['pockets']
        for pocket in pockets:
            center = pocket['center']
            half_width = pocket['half_width']
            strength = pocket['strength']
            
            start = max(0, center - half_width)
            end = min(N - 1, center + half_width)
            c_arr[start:end] = strength
    
    elif c_type == 'gradient':
        # For gradient types, apply gradient function
        gradient_params = coupling_config['gradient_params']
        center = gradient_params['center']
        spread = gradient_params['spread']
        start_val = gradient_params['start_value']
        end_val = gradient_params['end_value']
        
        # Create linear gradient
        for i in range(N - 1):
            distance = abs(i - center)
            if distance < spread:
                t = distance / spread
                c_arr[i] = start_val + (end_val - start_val) * t
    
    return c_arr


def create_theta_array(thermo_config: Dict[str, Any], N: int) -> np.ndarray:
    """
    Create a theta (threshold) array from profile configuration.
    
    Args:
        thermo_config: Thermodynamics configuration from profile
        N: Number of nodes in the network
        
    Returns:
        Theta array of length N
    """
    theta_type = thermo_config.get('theta_type', 'uniform')
    theta_value = thermo_config['theta']
    
    if theta_type == 'uniform':
        return np.ones(N) * theta_value
    else:
        # Could add other theta distributions here
        return np.ones(N) * theta_value


def run_profile(profile_name: str, 
                profile_path: str = "observer_profiles.json",
                master_seed: Optional[int] = None,
                verbose: bool = True) -> Dict[str, Any]:
    """
    Run a simulation using a named observer profile.
    
    Args:
        profile_name: Name of the profile to execute
        profile_path: Path to the observer profiles JSON file
        master_seed: Optional random seed (overrides default)
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing:
            - profile: The profile configuration used
            - counts: Joint count statistics array
            - I: Mutual information array
            - tau: Persistence times array
            - params: Extracted parameters for reference
    """
    profile = get_profile(profile_name, profile_path)
    
    if verbose:
        print(f"Running profile: {profile_name}")
        print(f"Description: {profile['description']}")
    
    # Extract parameters
    net_params = profile['parameters']['network']
    coupling_config = profile['parameters']['coupling']
    thermo_params = profile['parameters']['thermodynamics']
    sim_params = profile['parameters']['simulation']
    
    N = net_params['N']
    source_j = net_params['source_j']
    
    # Create arrays
    c_arr = create_coupling_array(coupling_config, N)
    theta_arr = create_theta_array(thermo_params, N)
    beta = thermo_params['beta']
    
    # Simulation parameters
    M = sim_params['M']
    T = sim_params['T']
    threshold = sim_params['threshold']
    continuous_source = sim_params.get('continuous_source', False)
    
    # Use provided seed or default
    seed = master_seed if master_seed is not None else DEFAULT_RANDOM_SEED
    
    if verbose:
        print(f"Network: N={N}, source={source_j}")
        print(f"Ensemble: M={M}, timesteps={T}")
        print(f"Beta={beta}, threshold={threshold}")
    
    # Run simulation
    counts = sim.run_simulation_chunked(
        N=N,
        c_arr=c_arr,
        beta=beta,
        theta_arr=theta_arr,
        M=M,
        T=T,
        source_j=source_j,
        master_seed=seed,
        continuous_source=continuous_source
    )
    
    # Compute derived quantities
    I = sim.compute_mi_from_counts(counts, M)
    tau = sim.compute_tau(I, threshold=threshold)
    
    if verbose:
        print(f"Simulation complete!")
        print(f"Mean tau: {tau.mean():.2f}")
        print(f"Max tau: {tau.max()}")
        print(f"Min tau: {tau.min()}")
    
    return {
        'profile': profile,
        'counts': counts,
        'I': I,
        'tau': tau,
        'params': {
            'N': N,
            'M': M,
            'T': T,
            'source_j': source_j,
            'beta': beta,
            'threshold': threshold,
            'c_arr': c_arr,
            'theta_arr': theta_arr
        }
    }


def compare_profiles(profile_names: List[str],
                    profile_path: str = "observer_profiles.json",
                    master_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run and compare multiple observer profiles.
    
    Args:
        profile_names: List of profile names to compare
        profile_path: Path to the observer profiles JSON file
        master_seed: Optional random seed for reproducibility
        
    Returns:
        Dictionary mapping profile names to their results
    """
    results = {}
    
    for profile_name in profile_names:
        print(f"\n{'='*60}")
        result = run_profile(profile_name, profile_path, master_seed, verbose=True)
        results[profile_name] = result
        print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Available observer profiles:")
    print("=" * 60)
    
    profiles = list_profiles()
    for i, name in enumerate(profiles, 1):
        profile = get_profile(name)
        print(f"{i:2d}. {name:30s} [{profile['category']}]")
        print(f"    {profile['description'][:70]}...")
    
    print("\n" + "=" * 60)
    print("\nRunning example profile: baseline_homogeneous\n")
    
    result = run_profile("baseline_homogeneous", verbose=True)
    
    print("\nExample complete! Results contain:")
    for key in result.keys():
        if key != 'profile':
            print(f"  - {key}")
