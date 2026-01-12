#!/usr/bin/env python3
"""
Example usage of observer profiles for comparative scalar modeling.

This script demonstrates how to use the observer profile system to:
1. List and explore available profiles
2. Run individual profiles
3. Compare multiple perspectives
4. Analyze and visualize results
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from profile_loader import (
    list_profiles, 
    get_profile, 
    run_profile, 
    compare_profiles
)


def explore_profiles():
    """Explore available observer profiles."""
    print("=" * 70)
    print("OBSERVER PROFILES CATALOG")
    print("=" * 70)
    
    # Get all profiles organized by category
    all_profiles = list_profiles()
    categories = {}
    
    for name in all_profiles:
        profile = get_profile(name)
        cat = profile['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(profile)
    
    # Display by category
    for category, profiles in sorted(categories.items()):
        print(f"\n{category.upper()} PROFILES:")
        print("-" * 70)
        for profile in profiles:
            print(f"\n  • {profile['name']}")
            print(f"    {profile['description']}")
            print(f"    Expected: {profile['expected_behavior']}")
            print(f"    Use cases: {', '.join(profile['use_cases'])}")
    
    print("\n" + "=" * 70)


def run_single_example():
    """Run a single profile and display results."""
    print("\n" + "=" * 70)
    print("SINGLE PROFILE EXAMPLE")
    print("=" * 70 + "\n")
    
    profile_name = "single_pocket_strong"
    result = run_profile(profile_name, verbose=True)
    
    # Extract results
    tau = result['tau']
    I = result['I']
    params = result['params']
    
    print(f"\nTau statistics:")
    print(f"  Mean: {tau.mean():.3f}")
    print(f"  Std:  {tau.std():.3f}")
    print(f"  Min:  {tau.min()}")
    print(f"  Max:  {tau.max()}")
    
    # Identify pocket region
    profile = result['profile']
    pocket = profile['parameters']['coupling']['pockets'][0]
    center = pocket['center']
    half_width = pocket['half_width']
    pocket_nodes = range(max(0, center - half_width), 
                        min(params['N'], center + half_width))
    
    print(f"\nInside pocket (nodes {list(pocket_nodes)}):")
    print(f"  Mean tau: {tau[list(pocket_nodes)].mean():.3f}")
    
    other_nodes = [i for i in range(params['N']) if i not in pocket_nodes]
    print(f"Outside pocket:")
    print(f"  Mean tau: {tau[other_nodes].mean():.3f}")
    
    return result


def compare_pocket_strengths():
    """Compare different pocket strength configurations."""
    print("\n" + "=" * 70)
    print("POCKET STRENGTH COMPARISON")
    print("=" * 70 + "\n")
    
    profiles_to_compare = [
        "baseline_homogeneous",
        "single_pocket_weak",
        "single_pocket_strong"
    ]
    
    results = compare_profiles(profiles_to_compare, master_seed=42)
    
    # Analyze results
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    for name, result in results.items():
        tau = result['tau']
        profile = result['profile']
        
        print(f"\n{name}:")
        print(f"  Mean tau: {tau.mean():.3f}")
        print(f"  Max tau:  {tau.max()}")
        
        # Get pocket info if applicable
        coupling = profile['parameters']['coupling']
        if coupling['type'] != 'uniform':
            pocket = coupling['pockets'][0]
            print(f"  Pocket strength: {pocket['strength']}")
    
    return results


def compare_temperature_regimes():
    """Compare high vs low temperature observers."""
    print("\n" + "=" * 70)
    print("TEMPERATURE REGIME COMPARISON")
    print("=" * 70 + "\n")
    
    profiles_to_compare = [
        "high_temperature_observer",
        "single_pocket_strong",  # Medium temperature
        "low_temperature_observer"
    ]
    
    results = compare_profiles(profiles_to_compare, master_seed=42)
    
    print("\n" + "=" * 70)
    print("TEMPERATURE EFFECTS SUMMARY")
    print("=" * 70)
    
    for name, result in results.items():
        tau = result['tau']
        beta = result['params']['beta']
        
        print(f"\n{name} (β={beta}):")
        print(f"  Mean tau: {tau.mean():.3f}")
        print(f"  Max tau:  {tau.max()}")
        print(f"  Temperature interpretation: {'high noise' if beta < 1 else 'low noise' if beta > 3 else 'moderate'}")
    
    return results


def demonstrate_profile_usage():
    """Demonstrate basic profile usage patterns."""
    print("\n" + "=" * 70)
    print("PROFILE USAGE PATTERNS")
    print("=" * 70 + "\n")
    
    # Pattern 1: Quick exploration
    print("Pattern 1: Quick exploration with list_profiles()")
    baseline_profiles = list_profiles(category='baseline')
    print(f"  Baseline profiles: {baseline_profiles}")
    
    pocket_profiles = list_profiles(category='single_pocket')
    print(f"  Single pocket profiles: {pocket_profiles}")
    
    # Pattern 2: Inspect before running
    print("\nPattern 2: Inspect profile details before running")
    profile = get_profile("dual_pocket_symmetric")
    print(f"  Profile: {profile['name']}")
    print(f"  Network size: {profile['parameters']['network']['N']} nodes")
    print(f"  Ensemble size: {profile['parameters']['simulation']['M']} members")
    print(f"  Number of pockets: {len(profile['parameters']['coupling']['pockets'])}")
    
    # Pattern 3: Run with custom seed for reproducibility
    print("\nPattern 3: Reproducible runs with custom seed")
    result1 = run_profile("baseline_homogeneous", master_seed=12345, verbose=False)
    result2 = run_profile("baseline_homogeneous", master_seed=12345, verbose=False)
    identical = np.allclose(result1['I'], result2['I'])
    print(f"  Results with same seed are identical: {identical}")


def main():
    """Main demonstration function."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "OBSERVER PROFILES DEMONSTRATION" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # 1. Explore all available profiles
    explore_profiles()
    
    # 2. Run a single profile example
    run_single_example()
    
    # 3. Compare pocket strengths
    compare_pocket_strengths()
    
    # 4. Compare temperature regimes
    compare_temperature_regimes()
    
    # 5. Demonstrate usage patterns
    demonstrate_profile_usage()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - observer_profiles.json: Profile definitions")
    print("  - profile_loader.py: Profile loading utilities")
    print("  - README.md: Documentation and usage guide")
    print()


if __name__ == "__main__":
    main()
