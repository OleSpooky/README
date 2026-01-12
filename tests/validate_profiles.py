#!/usr/bin/env python3
"""
Validation tests for observer profiles.

This script checks that all observer profiles:
1. Have valid structure and required fields
2. Can be loaded successfully
3. Produce valid simulation results
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
from pathlib import Path
from profile_loader import load_profiles, run_profile


# Numerical tolerance for mutual information values
# Small negative values can occur due to floating-point precision in probability calculations
MI_NEGATIVE_TOLERANCE = -0.01


def validate_profile_structure(profile):
    """Validate that a profile has all required fields."""
    required_fields = ['name', 'description', 'category', 'parameters', 'expected_behavior', 'use_cases']
    
    for field in required_fields:
        if field not in profile:
            return False, f"Missing required field: {field}"
    
    # Check parameters structure
    params = profile['parameters']
    required_param_sections = ['network', 'coupling', 'thermodynamics', 'simulation']
    
    for section in required_param_sections:
        if section not in params:
            return False, f"Missing parameter section: {section}"
    
    # Check network parameters
    network = params['network']
    if 'N' not in network or 'source_j' not in network:
        return False, "Network parameters incomplete"
    
    # Check simulation parameters
    simulation = params['simulation']
    if 'M' not in simulation or 'T' not in simulation:
        return False, "Simulation parameters incomplete"
    
    return True, "OK"


def validate_profile_values(profile):
    """Validate that profile parameter values are reasonable."""
    params = profile['parameters']
    
    # Check network values
    N = params['network']['N']
    source_j = params['network']['source_j']
    
    if N < 2:
        return False, f"N={N} is too small (minimum 2)"
    
    if source_j < 0 or source_j >= N:
        return False, f"source_j={source_j} is out of bounds for N={N}"
    
    # Check simulation values
    M = params['simulation']['M']
    T = params['simulation']['T']
    
    if M < 1:
        return False, f"M={M} is too small (minimum 1)"
    
    if T < 1:
        return False, f"T={T} is too small (minimum 1)"
    
    # Check thermodynamic values
    beta = params['thermodynamics']['beta']
    if beta <= 0:
        return False, f"beta={beta} must be positive"
    
    return True, "OK"


def validate_profile_execution(profile_name, quick_test=True):
    """Validate that a profile can be executed successfully."""
    try:
        # Run with consistent random seed for reproducibility
        # Note: Profile parameters (M, T, N) are defined in the profile itself
        # and cannot be easily overridden without modifying the profile
        result = run_profile(profile_name, master_seed=1, verbose=False)
        
        # Check that results have expected structure
        required_keys = ['counts', 'I', 'tau', 'params']
        for key in required_keys:
            if key not in result:
                return False, f"Missing result key: {key}"
        
        # Check that arrays have reasonable shapes
        tau = result['tau']
        I = result['I']
        params = result['params']
        
        if len(tau) != params['N']:
            return False, f"tau length {len(tau)} doesn't match N={params['N']}"
        
        if I.shape[1] != params['N']:
            return False, f"I shape {I.shape} doesn't match N={params['N']}"
        
        # Check that values are in reasonable ranges
        if not np.all(np.isfinite(tau)):
            return False, "tau contains non-finite values"
        
        if not np.all(tau >= 0):
            return False, "tau contains negative values"
        
        if not np.all(np.isfinite(I)):
            return False, "I contains non-finite values"
        
        if not np.all(I >= MI_NEGATIVE_TOLERANCE):
            return False, "I contains significantly negative values"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Execution failed: {str(e)}"


def run_validation():
    """Run all validation tests."""
    print("=" * 70)
    print("OBSERVER PROFILES VALIDATION")
    print("=" * 70)
    
    # Load profiles
    try:
        data = load_profiles()
        profiles = data['profiles']
        print(f"\n✓ Successfully loaded {len(profiles)} profiles")
    except Exception as e:
        print(f"\n✗ Failed to load profiles: {e}")
        return False
    
    # Validate each profile
    all_passed = True
    
    print("\nValidating profile structure and values:")
    print("-" * 70)
    
    for profile in profiles:
        name = profile['name']
        
        # Structure validation
        valid, msg = validate_profile_structure(profile)
        if not valid:
            print(f"✗ {name}: Structure validation failed - {msg}")
            all_passed = False
            continue
        
        # Value validation
        valid, msg = validate_profile_values(profile)
        if not valid:
            print(f"✗ {name}: Value validation failed - {msg}")
            all_passed = False
            continue
        
        print(f"✓ {name}: Structure and values OK")
    
    # Test execution for a subset of profiles
    print("\nValidating profile execution (sample):")
    print("-" * 70)
    
    # Test a representative sample from each category
    test_profiles = [
        'baseline_homogeneous',
        'single_pocket_strong',
        'high_temperature_observer'
    ]
    
    for profile_name in test_profiles:
        valid, msg = validate_profile_execution(profile_name, quick_test=True)
        if not valid:
            print(f"✗ {profile_name}: Execution failed - {msg}")
            all_passed = False
        else:
            print(f"✓ {profile_name}: Execution successful")
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print("✗ SOME VALIDATIONS FAILED")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = run_validation()
    exit(0 if success else 1)
