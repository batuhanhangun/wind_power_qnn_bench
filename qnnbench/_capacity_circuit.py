"""Capacity-parameterized reverse_linear QNN circuit.

This is the reverse_linear training circuit of ``src/reference_circuit.py``,
refactored so the number of ansatz repetitions is a *parameter* rather than a
module-level constant. The gate sequence is otherwise byte-for-byte identical to
the deployed model, so ``make_train_circuit(3)`` reproduces the reference circuit
exactly (the correctness gate asserts agreement within 1e-10).

Gate sequence (RealAmplitudes, reverse_linear), for a given ``reps``:
  1. Feature map ZFeatureMap(4, reps=1): Hadamard then RZ(2*x) on each qubit.
  2. Initial RY layer (no entangling before the first layer).
  3. ``reps`` repetitions of [CNOT pattern (2,3),(1,2),(0,1), then RY layer].
  4. Measurement of PauliZ on all four qubits as a tensor product.

CNOTs are unparameterized, so the trainable parameter count is exactly
4*(reps+1) and the CNOT count is exactly 3*reps (reverse_linear, 3 per rep).
"""

import sys
from pathlib import Path

import pennylane as qml

from qnnbench import config

NUM_QUBITS = config.QNN_CONFIG["num_qubits"]   # 4
CNOT_PAIRS = config.ENTANGLEMENT_PAIRS         # [(2,3),(1,2),(0,1)]


def weight_shape(reps):
    """2D weight shape (reps+1, num_qubits) consumed by the ansatz."""
    return (reps + 1, NUM_QUBITS)


def num_params(reps):
    """Trainable parameter count: 4*(reps+1)."""
    return (reps + 1) * NUM_QUBITS


def num_cnots(reps):
    """CNOT count: len(reverse_linear pattern) * reps = 3*reps."""
    return len(CNOT_PAIRS) * reps


# --- Gate-sequence primitives (identical to reference_circuit) ---

def _apply_feature_map(x):
    """ZFeatureMap: H + RZ(2*x) per qubit."""
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(2.0 * x[i], wires=i)


def _apply_ansatz(weights, reps):
    """RealAmplitudes(reverse_linear): initial RY layer, then per-rep CNOTs + RY."""
    # Initial RY layer (no entanglement before the first layer).
    for i in range(NUM_QUBITS):
        qml.RY(weights[0, i], wires=i)

    # reps repetitions of [reverse_linear CNOT pattern + RY layer].
    for rep in range(1, reps + 1):
        for (ctrl, tgt) in CNOT_PAIRS:
            qml.CNOT(wires=[ctrl, tgt])
        for i in range(NUM_QUBITS):
            qml.RY(weights[rep, i], wires=i)


def _measurement():
    """PauliZ tensor product on all four qubits."""
    return qml.expval(
        qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
    )


# --- Circuit factory (one QNode per reps value, cached) ---

_CIRCUIT_CACHE = {}


def make_train_circuit(reps):
    """Return the noise-free lightning.qubit QNode for the given reps.

    Cached so repeated calls for the same reps return the identical QNode
    (and re-use its compiled device). Uses the autograd interface with adjoint
    differentiation, exactly as the reference reverse_linear training circuit.

    Args:
        reps: Number of ansatz repetitions (capacity axis).

    Returns:
        Callable QNode: circuit(weights_2d, x) -> scalar expectation value,
        where weights_2d has shape (reps+1, 4).
    """
    if reps in _CIRCUIT_CACHE:
        return _CIRCUIT_CACHE[reps]

    dev = qml.device(config.QNN_CONFIG["device_train"], wires=NUM_QUBITS)

    @qml.qnode(dev, interface="autograd",
               diff_method=config.QNN_CONFIG["diff_method"])
    def circuit(weights, x):
        _apply_feature_map(x)
        _apply_ansatz(weights, reps)
        return _measurement()

    _CIRCUIT_CACHE[reps] = circuit
    return circuit
