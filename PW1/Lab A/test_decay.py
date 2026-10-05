import pytest
import numpy as np
import decay

def test_one_step():
    """Example test given by the lab manual."""
    n = decay.simulate(100, 0.1, steps=1)
    assert len(n) == 2
    assert n[0] == 100
    assert n[1] <= 100

def test_negative_rate_raises_value_error():
    """TODO 1: Checks that simulate raises a ValueError for a negative rate."""
    with pytest.raises(ValueError):
        decay.simulate(100, -0.4)

def test_simulation_average_close_to_N0_half():
    """TODO 2: Checks that the simulation's average over many seeds matches the physical law."""
    N0 = 1000
    rate = 0.5
    steps = 1
    dt = 0.05  # The time-step size used in the simulation
    results = []
    
    # Run the simulation across 100 different random seeds
    for seed in range(100):
        n = decay.simulate(N0, rate, steps=steps, seed=seed)
        results.append(n[-1]) # Grab the value at the final step
        
    average_remaining = np.mean(results)
    
    # Mathematical decay formula: N(t) = N0 * e^(-lambda * t)
    expected_remaining = N0 * np.exp(-rate * steps * dt)
    
    # Verify the average matches the theory within a small tolerance
    assert average_remaining == pytest.approx(expected_remaining, abs=10)
