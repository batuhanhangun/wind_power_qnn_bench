"""Mandatory correctness gate.

Asserts that the parameterized circuit with ``entanglement="circular"`` returns an
expectation value identical, within 1e-10, to the reference circuit in
``src/run_qnn.py`` for the same weights and input vector. This guarantees the
candidate machinery reproduces the original incumbent (QNN-3) gate-for-gate.

Run standalone BEFORE submitting the array job:

    python -m tests.test_circuit_equivalence

It prints PASS and exits 0 on success, or prints FAIL and exits 1 on divergence,
so it can gate the pipeline in a shell (`&&`) chain.
"""

import sys
from pathlib import Path

import numpy as np
from pennylane import numpy as pnp

# Allow `from src...` imports from the project root.
from qnnbench.circuits import make_circuit, WEIGHT_SHAPE, NUM_QUBITS
from tests.original.circular import circuit as reference_circuit

TOL = 1e-10
N_TRIALS = 25


def _gate_test_circular_matches_reference():
    """Return the maximum |Δ| over random trials; assert it is below TOL."""
    rng = np.random.RandomState(12345)
    candidate = make_circuit("circular")

    max_diff = 0.0
    for _ in range(N_TRIALS):
        weights = pnp.array(
            rng.uniform(-np.pi, np.pi, size=WEIGHT_SHAPE), requires_grad=False
        )
        x = pnp.array(
            rng.uniform(-2.0, 2.0, size=NUM_QUBITS), requires_grad=False
        )
        ref = float(reference_circuit(weights, x))
        cand = float(candidate(weights, x))
        max_diff = max(max_diff, abs(ref - cand))

    assert max_diff < TOL, (
        f"circular diverges from the reference circuit: "
        f"max|Δ|={max_diff:.3e} >= TOL={TOL:.0e}"
    )
    return max_diff



def test_circular_matches_reference():
    _gate_test_circular_matches_reference()
