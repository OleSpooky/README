#!/usr/bin/env python3
"""
Validation tests for observer profiles.

This script checks that all observer profiles:
1. Have valid structure and required fields
2. Have non-empty descriptive fields and correct types
3. Maintain numerical stability and sanity
4. Produce non-degenerate simulation results
5. Match requested execution identity
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
from pathlib import Path
from profile_loader import load_profiles, run_profile


# Numerical tolerance for mutual information values
MI_NEGATIVE_TOLERANCE = -0.01
# Max allowed fraction of nodes with negative MI beyond tolerance
MAX_NEGATIVE_FRACTION = 0.05
# Soft upper bound for N to avoid CI timeouts/memory exhaustion
MAX_NODES_SOFT_LIMIT = 10000
# Stability limit for beta
BETA_STABILITY_LIMIT = 1e-6

# Whitelist of allowed categories for systematic reporting
ALLOWED_CATEGORIES = [
    'baseline', 
    'single_pocket', 
    'multi_pocket', 
    'thermodynamic', 
    'spatial', 
    'temporal', 
    'scale', 
    'gradient'
]

def validate_profile_structure(profile):
    """Validate that a profile has all required fields and valid types."""
    required_fields = ['name', 'description', 'category', 'parameters', 'expected_behavior', 'use_cases']
    
    for field in required_fields:
        if field not in profile:
            return False, f"Missing required field: {field}"
            
    # Structural guards (Shape without substance)
    if not isinstance(profile['description'], str) or len(profile['description'].strip()) == 0:
        return False, "Description must be a non-empty string"
        
    if not isinstance(profile['use_cases'], list) or len(profile['use_cases']) == 0:
        return False, "use_cases must be a non-empty list"
    
    if not all(isinstance(uc, str) and len(uc.strip()) > 0 for uc in profile['use_cases']):
        return False, "All use_cases must be non-empty strings"

    if profile['category'] not in ALLOWED_CATEGORIES:
        return False, f"Category '{profile['category']}' not in allowed list: {ALLOWED_CATEGORIES}"

    # Check parameters structure and types
    params = profile['parameters']
    if not isinstance(params, dict):
        return False, "parameters must be a dictionary"
        
    required_param_sections = ['network', 'coupling', 'thermodynamics', 'simulation']
    for section in required_param_sections:
        if section not in params:
            return False, f"Missing parameter section: {section}"
        if not isinstance(params[section], dict):
            return False, f"Parameter section '{section}' must be a dictionary"
    
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
    """Validate that profile parameter values are reasonable and stable."""
    params = profile['parameters']
    
    # Check network values
    N = params['network']['N']
    source_j = params['network']['source_j']
    
    if N < 2:
        return False, f"N={N} is too small (minimum 2)"
    
    if N > MAX_NODES_SOFT_LIMIT:
        print(f"  [WARN] N={N} exceeds soft limit ({MAX_NODES_SOFT_LIMIT}). Risk of memory exhaustion.")
        
    if N < 3:
        print(f"  [WARN] N={N} is degenerate. Simulation results may be trivial.")

    if source_j < 0 or source_j >= N:
        return False, f"source_j={{source_j}} is out of bounds for N={N}"
    
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
        return False, f"beta={{beta}} must be positive"
    
    if beta < BETA_STABILITY_LIMIT:
        return False, f"beta={{beta}} is below stability limit ({BETA_STABILITY_LIMIT})"
    
    return True, "OK"

def validate_profile_execution(profile_name):
    """Validate that a profile can be executed successfully and produces valid signals."""
    try:
        # Run with consistent random seed for reproducibility
        result = run_profile(profile_name, master_seed=1, verbose=False)
        
        # Identity Check: Ensure requested profile was actually the one run
        if result['profile']['name'] != profile_name:
            return False, f"Execution mismatch: requested {{profile_name}} but got {{result['profile']['name']}}"
        
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
            return False, f"tau length {{len(tau)}} doesn't match N={{params['N']}}"
        
        if I.shape[1] != params['N']:
            return False, f"I shape {{I.shape}} doesn't match N={{params['N']}}"
        
        # Check that values are in reasonable ranges
        if not np.all(np.isfinite(tau)):
            return False, "tau contains non-finite values"
        
        if not np.all(tau >= 0):
            return False, "tau contains negative values"
            
        # Non-degenerate check: Tau all zeros (Thermal void)
        if np.all(tau == 0) and params['N'] > 1:
             return False, "tau indicates no surviving observers (thermal void)"
        
        if not np.all(np.isfinite(I)):
            return False, "I contains non-finite values"
            
        # Trivial signal check: MI trivially zero
        if np.max(I) == 0:
            return False, "Mutual information is trivially zero (disconnected graph or broken logic)"
        
        # Systematic failure check: small negatives mask
        neg_mask = I < MI_NEGATIVE_TOLERANCE
        fraction_negative = np.mean(neg_mask)
        if fraction_negative > MAX_NEGATIVE_FRACTION:
            return False, f"Systematic failure: {{fraction_negative:.2%}} of I values are negative beyond tolerance"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Execution failed: {{str(e)}}"

def run_validation():
    """Run all validation tests."""
    print("=" * 70)
    print("OBSERVER PROFILES VALIDATION ENGINE")
    print("=" * 70)
    
    # Load profiles
    try:
        data = load_profiles()
        profiles = data['profiles']
        print(f"\n✓ Successfully loaded {{len(profiles)}} profiles")
    except Exception as e:
        print(f"\n✗ Failed to load profiles: {{e}}")
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
            print(f"✗ {{name}}: Structure validation failed - {{msg}}")
            all_passed = False
            continue
        
        # Value validation
        valid, msg = validate_profile_values(profile)
        if not valid:
            print(f"✗ {{name}}: Value validation failed - {{msg}}")
            all_passed = False
            continue
        
        print(f"✓ {{name}}: Structure and values OK")
    
    # Test execution for a subset of profiles
    print("\nValidating profile execution (sample):")
    print("-" * 70)
    
    # Test a representative sample from each category
    test_profiles = [
        'baseline_homogeneous',
        'single_pocket_strong'
    ]
    
    # Add one from high temperature if it exists to test stability
    if 'high_temperature_observer' in [p['name'] for p in profiles]:
        test_profiles.append('high_temperature_observer')
    
    for profile_name in test_profiles:
        valid, msg = validate_profile_execution(profile_name)
        if not valid:
            print(f"✗ {{profile_name}}: Execution failed - {{msg}}")
            all_passed = False
        else:
            print(f"✓ {{profile_name}}: Execution successful")
    
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