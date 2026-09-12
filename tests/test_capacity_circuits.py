"""Mandatory correctness gate — run and PASS before the full capacity sweep.

Asserts:
  (a) At reps=3 the trainable parameter count is exactly 16 (gradient length 16)
      and the CNOT count is exactly 9.
  (b) The capacity formula holds across the whole grid: 4*(reps+1) params and
      3*reps CNOTs for reps in {1,2,3,4,5}, i.e. params {8,12,16,20,24} and
      CNOTs {3,6,9,12,15}.
  (c) Reference agreement: the parameterized circuit at reps=3 reproduces the
      provided reverse_linear reference circuit within 1e-10 for identical
      weights and inputs. Any divergence means the swept circuit drifted from
      the deployed gate sequence.

Run standalone (gates the pipeline in a shell `&&` chain):

    python -m tests.test_capacity

Prints PASS and exits 0 on success; prints FAIL and exits 1 on any mismatch.
"""

import sys
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qnnbench import config
from qnnbench.circuits import (
    make_train_circuit,
    weight_shape,
    num_params,
    num_cnots,
    NUM_QUBITS,
)
from qnnbench.circuits import circuit as reference_circuit

TOL = 1e-10
N_TRIALS = 25

EXPECTED_PARAMS = {1: 8, 2: 12, 3: 16, 4: 20, 5: 24}
EXPECTED_CNOTS = {1: 3, 2: 6, 3: 9, 4: 12, 5: 15}


def _gate_test_deployed_counts():
    """(a) reps=3 -> exactly 16 trainable params (grad length) and 9 CNOTs."""
    reps = config.DEPLOYED_REPS
    assert num_params(reps) == 16, f"num_params(3)={num_params(reps)}, expected 16"

    circuit = make_train_circuit(reps)
    shape = weight_shape(reps)

    def flat_cost(weights_flat, x):
        return circuit(weights_flat.reshape(shape), x)

    rng = np.random.RandomState(7)
    w = pnp.array(rng.uniform(-np.pi, np.pi, size=num_params(reps)),
                  requires_grad=True)
    x = pnp.array(rng.uniform(-2.0, 2.0, size=NUM_QUBITS), requires_grad=False)

    grad = qml.grad(flat_cost, argnums=0)(w, x)
    n_trainable = int(np.asarray(grad).size)
    assert n_trainable == 16, f"Trainable param count = {n_trainable}, expected 16"

    specs = qml.specs(circuit)(w.reshape(shape), x)
    gate_types = dict(specs["resources"].gate_types)
    n_cnot = gate_types.get("CNOT", 0)
    assert n_cnot == 9, f"CNOT count at reps=3 = {n_cnot}, expected 9"
    return n_trainable, n_cnot


def _gate_test_capacity_formula():
    """(b) 4*(reps+1) params and 3*reps CNOTs across the whole grid."""
    for reps in config.REPS_GRID:
        p = num_params(reps)
        c = num_cnots(reps)
        assert p == EXPECTED_PARAMS[reps], (
            f"reps={reps}: num_params={p}, expected {EXPECTED_PARAMS[reps]}"
        )
        assert c == EXPECTED_CNOTS[reps], (
            f"reps={reps}: num_cnots={c}, expected {EXPECTED_CNOTS[reps]}"
        )

        # Cross-check the CNOT count against the actually recorded tape.
        circuit = make_train_circuit(reps)
        rng = np.random.RandomState(100 + reps)
        w = pnp.array(rng.uniform(-np.pi, np.pi, size=weight_shape(reps)),
                      requires_grad=False)
        x = pnp.array(rng.uniform(-2.0, 2.0, size=NUM_QUBITS), requires_grad=False)
        specs = qml.specs(circuit)(w, x)
        n_cnot = dict(specs["resources"].gate_types).get("CNOT", 0)
        assert n_cnot == EXPECTED_CNOTS[reps], (
            f"reps={reps}: taped CNOTs={n_cnot}, expected {EXPECTED_CNOTS[reps]}"
        )
    return {r: (EXPECTED_PARAMS[r], EXPECTED_CNOTS[r]) for r in config.REPS_GRID}


def _gate_test_reference_agreement():
    """(c) reps=3 circuit matches the provided reverse_linear reference < 1e-10."""
    reps = config.DEPLOYED_REPS
    circuit = make_train_circuit(reps)
    shape = weight_shape(reps)

    rng = np.random.RandomState(2024)
    max_diff = 0.0
    for _ in range(N_TRIALS):
        w = pnp.array(rng.uniform(-np.pi, np.pi, size=shape), requires_grad=False)
        x = pnp.array(rng.uniform(-2.0, 2.0, size=NUM_QUBITS), requires_grad=False)
        mine = float(circuit(w, x))
        ref = float(reference_circuit(w, x))
        max_diff = max(max_diff, abs(mine - ref))

    assert max_diff < TOL, (
        f"reps=3 vs reference diverges: max|d|={max_diff:.3e} >= TOL={TOL:.0e}"
    )
    return max_diff



def test_deployed_counts():
    _gate_test_deployed_counts()

def test_capacity_formula():
    _gate_test_capacity_formula()

def test_reference_agreement():
    _gate_test_reference_agreement()
