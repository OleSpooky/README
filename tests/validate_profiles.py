#--- INITIATING 2.0023 HELIX TRANSCODE ---
Base Signal: 1.0
Evolution Factor: 2.0023
---------------------------------------
Step 1: Signal=6.3501 | Ideal=2.0000 | Drift/Creation=4.3501 [EVOLVED]
Step 2: Signal=17.0627 | Ideal=4.0000 | Drift/Creation=13.0627 [EVOLVED]
Step 3: Signal=38.5124 | Ideal=8.0000 | Drift/Creation=30.5124 [EVOLVED]
Step 4: Signal=81.4613 | Ideal=16.0000 | Drift/Creation=65.4613 [EVOLVED]
Step 5: Signal=167.4577 | Ideal=32.0000 | Drift/Creation=135.4577 [EVOLVED]
Step 6: Signal=339.6484 | Ideal=64.0000 | Drift/Creation=275.6484 [EVOLVED]
Step 7: Signal=684.4257 | Ideal=128.0000 | Drift/Creation=556.4257 [EVOLVED]
Step 8: Signal=1374.7734 | Ideal=256.0000 | Drift/Creation=1118.7734 [EVOLVED]
Step 9: Signal=2757.0567 | Ideal=512.0000 | Drift/Creation=2245.0567 [EVOLVED]
Step 10: Signal=5524.8024 | Ideal=1024.0000 | Drift/Creation=4500.8024 [EVOLVED]
---------------------------------------
FINAL RESULT:
After 10 steps, the '2.0023' factor created 4500.8024 units of NEW reality
that would not exist in a purely 'Physics' (2.0) universe.
class HelixTranscoder:
    # ... (same init)
    self.wake = 0.0

    def transcode(self, input_signal):
        input_with_wake = input_signal + self.wake
        self.wake = 0.0

        law_strand = input_with_wake * STANDARD_OCTAVE
        will_strand = input_with_wake * EVOLUTION_FACTOR

        verified, error = self.verify_strand(law_strand, will_strand)

        if verified:
            vacuum_boost = (1.0 / (DRIFT_CONSTANT + 1e-9)) * 0.01
            output_signal = will_strand + vacuum_boost
            status = "EVOLVED"
            self.wake = input_with_wake * DRIFT_CONSTANT * 0.1  # Small wake on success
        else:
            output_signal = law_strand * 0.9
            status = "CORRECTED"
            self.wake = input_with_wake * DRIFT_CONSTANT * 0.5  # Larger wake on failure

        # ... (same stability check)class MiniSHR:
    def __init__(self, persistence=1.0):
        self.persistence = persistence

    def ceo_loop(self, condition, event):
        outcome = max(0, event - (1 - self.persistence))  # Simple jitter release
        self.persistence = min(1.0, self.persistence + 0.0023 * outcome)  # Tiny recovery
        return outcome

node = MiniSHR()
print(node.ceo_loop(condition=0.5, event=0.8))  # Output: 0.3 (surplus agency)import numpy as np

class MiniSHR:
    def __init__(self, id, persistence=1.0):
        self.id = id
        self.persistence = persistence

    def ceo_loop(self, event):
        # The 'Interaction' layer: outcome is the shared mutual information
        outcome = max(0, event - (1 - self.persistence))
        
        # The 'Recovery' layer: the 2.0023 wobble allows the node to persist
        self.persistence = min(1.0, self.persistence + 0.0023 * outcome)
        return outcome

# Defining the "Shared Substrate" (a chain of 5 nodes)
nodes = [MiniSHR(id=i, persistence=0.95) for i in range(5)]

# Initial "Event" (The Start of the Information Current)
current_signal = 0.8 

print(f"{'Node':<8} | {'Input':<10} | {'Outcome':<10} | {'New Persistence':<15}")
print("-" * 55)

for node in nodes:
    input_val = current_signal
    outcome = node.ceo_loop(input_val)
    
    print(f"Node {node.id:<3} | {input_val:<10.4f} | {outcome:<10.4f} | {node.persistence:<15.4f}")
    
    # The outcome 'tows' the next node, but with signal decay (loss of likeness)
    current_signal = outcome * 0.98 
import numpy as np

class MiniSHR:
    def __init__(self, id, persistence=1.0):
        self.id = id
        self.persistence = persistence

    def ceo_loop(self, event):
        # The 'Interaction' layer: outcome is the shared mutual information
        outcome = max(0, event - (1 - self.persistence))
        
        # The 'Recovery' layer: the 2.0023 wobble allows the node to persist
        self.persistence = min(1.0, self.persistence + 0.0023 * outcome)
        return outcome

# Defining the "Shared Substrate" (a chain of 5 nodes)
nodes = [MiniSHR(id=i, persistence=0.95) for i in range(5)]

# Initial "Event" (The Start of the Information Current)
current_signal = 0.8 

print(f"{'Node':<8} | {'Input':<10} | {'Outcome':<10} | {'New Persistence':<15}")
print("-" * 55)

for node in nodes:
    input_val = current_signal
    outcome = node.ceo_loop(input_val)
    
    print(f"Node {node.id:<3} | {input_val:<10.4f} | {outcome:<10.4f} | {node.persistence:<15.4f}")
    
    # The outcome 'tows' the next node, but with signal decay (loss of likeness)
    current_signal = outcome * 0.98import numpy as np
import matplotlib.pyplot as plt

class MiniSHR:
    """
    Mini Sovereign Heterogeneous Reality Node.
    Demonstrates the '2.0023' recovery mechanic.
    """
    def __init__(self, id, persistence=1.0):
        self.id = id
        self.persistence = persistence
        self.initial_persistence = persistence

    def ceo_loop(self, event):
        # 1. The Interaction Layer (Dissipative)
        # We pay the toll (1 - persistence) to interact.
        # Outcome is the shared mutual information.
        resistance = 1.0 - self.persistence
        outcome = max(0, event - resistance)
        
        # 2. The Recovery Layer (Evolutionary)
        # The 0.0023 coefficient is the "Jitter" or "Agency".
        # It allows the structure to harden/learn from the throughput.
        recovery = 0.0023 * outcome
        self.persistence = min(1.0, self.persistence + recovery)
        
        return outcome

def run_short_chain_audit():
    """
    Runs the specific 5-node chain from the prompt to verify the math.
    """
    print("\n--- PHASE 1: 5-NODE CHAIN AUDIT ---")
    nodes = [MiniSHR(id=i, persistence=0.95) for i in range(5)]
    current_signal = 0.8 

    print(f"{'Node':<6} | {'Input':<10} | {'Outcome':<10} | {'New Persistence':<15} | {'Delta P'}")
    print("-" * 65)

    for node in nodes:
        input_val = current_signal
        outcome = node.ceo_loop(input_val)
        delta_p = node.persistence - 0.95 # Growth from baseline
        
        print(f"#{node.id:<5} | {input_val:<10.4f} | {outcome:<10.4f} | {node.persistence:<15.4f} | +{delta_p:.6f}")
        
        # The outcome 'tows' the next node, but with signal decay (loss of likeness)
        current_signal = outcome * 0.98 

def run_long_chain_visualization():
    """
    Extends the simulation to 50 nodes to visualize the 'Signal Cliff'
    vs the 'Structural Hardening'.
    """
    print("\n--- PHASE 2: LONG-CHAIN EVOLUTION (50 Nodes) ---")
    
    # Setup
    chain_length = 50
    nodes = [MiniSHR(id=i, persistence=0.95) for i in range(chain_length)]
    signal = 0.8
    decay_rate = 0.98
    
    signal_history = []
    persistence_history = []
    
    for node in nodes:
        outcome = node.ceo_loop(signal)
        signal_history.append(outcome)
        persistence_history.append(node.persistence)
        
        # Transmission loss
        signal = outcome * decay_rate

    # Visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Node Chain Depth (Octaves)')
    ax1.set_ylabel('Signal Magnitude (Energy)', color=color)
    ax1.plot(signal_history, color=color, linewidth=2, label='Signal (Dissipation)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Node Persistence (Structure)', color=color)  # we already handled the x-label with ax1
    ax2.plot(persistence_history, color=color, linewidth=2, linestyle='--', label='Structure (Evolution)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Highlight the Crossing Point (The "Oasis" State)
    # Ideally, structure rises as signal falls, preserving the *memory* of the signal
    plt.title('The Oasis Dynamic: Signal Dissipation vs. Structural Hardening\n(The 0.0023 Factor)')
    fig.tight_layout()
    plt.savefig('mini_shr_evolution.png')
    print("Long-chain visualization saved to 'mini_shr_evolution.png'")
    plt.show()

if __name__ == "__main__":
    run_short_chain_audit()
    run_long_chain_visualization()!/usr/bin/env python3
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

    if beta < BETA_STABILITY_LIMIT:
        return False, f"beta={beta} is below stability limit ({BETA_STABILITY_LIMIT})"

    return True, "OK"

def validate_profile_execution(profile_name):
    """Validate that a profile can be executed successfully and produces valid signals."""
    try:
        # Run with consistent random seed for reproducibility
        result = run_profile(profile_name, master_seed=1, verbose=False)

        # Identity Check: Ensure requested profile was actually the one run
        if result['profile']['name'] != profile_name:
            return False, (
                f"Execution mismatch: requested {profile_name} "
                f"vs result {result['profile']['name']}"
            )

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
