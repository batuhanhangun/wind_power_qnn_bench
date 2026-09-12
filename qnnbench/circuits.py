"""Shared reverse_linear QNN circuit — one definition for every QNode.

Copied verbatim from the reverse_linear rerun (the architecture the paper's
revision adopts), with ONE Q5 addition: ``make_trainable_noisy_circuit`` — a
density-matrix QNode that is differentiable, so the SAME noisy circuit the paper
evaluates can now also be TRAINED. The noise injection (DepolarizingChannel
placement) is defined once in the shared primitives below and reused by the
noise-free training circuit, the frozen-weight noisy evaluation circuit, and the
new trainable noisy circuit, so the three can never drift apart — which is the
precondition for a valid freeze-evaluate vs noise-aware comparison.

Gate sequence (RealAmplitudes, reverse_linear):
  1. Feature map ZFeatureMap(4, reps=1): Hadamard then RZ(2*x) on each qubit.
  2. Initial RY layer (no entangling before the first layer).
  3. ANSATZ_REPS repetitions of [CNOT pattern (2,3),(1,2),(0,1), then RY layer].
  4. Measurement of PauliZ on all four qubits as a tensor product.

CNOTs are unparameterized, so the model has exactly 16 trainable parameters;
the reverse_linear pattern uses 9 CNOTs (3 per rep).
"""

import sys
from pathlib import Path

import pennylane as qml

from qnnbench import config

NUM_QUBITS = config.QNN_CONFIG["num_qubits"]      # 4
ANSATZ_REPS = config.QNN_CONFIG["ansatz_reps"]    # 3
WEIGHT_SHAPE = (ANSATZ_REPS + 1, NUM_QUBITS)      # (4, 4)
NUM_PARAMS = WEIGHT_SHAPE[0] * WEIGHT_SHAPE[1]    # 16
CNOT_PAIRS = config.ENTANGLEMENT_PAIRS            # [(2,3),(1,2),(0,1)]


# --- Gate-sequence primitives (shared by noise-free and noisy circuits) ---

def _apply_feature_map(x, p=None):
    """ZFeatureMap: H + RZ(2*x) per qubit, optional depolarizing after each gate."""
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
        if p is not None:
            qml.DepolarizingChannel(p, wires=i)
        qml.RZ(2.0 * x[i], wires=i)
        if p is not None:
            qml.DepolarizingChannel(p, wires=i)


def _apply_ansatz(weights, p=None):
    """RealAmplitudes(reverse_linear): initial RY layer, then per-rep CNOTs + RY.

    When ``p`` is given, a single-qubit DepolarizingChannel(p) is inserted after
    every gate (after each RY; after BOTH qubits of each CNOT, independently),
    exactly matching the published noise harness.
    """
    # Initial RY layer (no entanglement before the first layer).
    for i in range(NUM_QUBITS):
        qml.RY(weights[0, i], wires=i)
        if p is not None:
            qml.DepolarizingChannel(p, wires=i)

    # reps repetitions of [reverse_linear CNOT pattern + RY layer].
    for rep in range(1, ANSATZ_REPS + 1):
        for (ctrl, tgt) in CNOT_PAIRS:
            qml.CNOT(wires=[ctrl, tgt])
            if p is not None:
                qml.DepolarizingChannel(p, wires=ctrl)
                qml.DepolarizingChannel(p, wires=tgt)
        for i in range(NUM_QUBITS):
            qml.RY(weights[rep, i], wires=i)
            if p is not None:
                qml.DepolarizingChannel(p, wires=i)


def _measurement():
    """PauliZ tensor product on all four qubits."""
    return qml.expval(
        qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
    )


# --- Noise-free training circuit (lightning.qubit, adjoint diff) ---

_dev_train = qml.device(config.QNN_CONFIG["device_train"], wires=NUM_QUBITS)


@qml.qnode(_dev_train, interface="autograd",
           diff_method=config.QNN_CONFIG["diff_method"])
def circuit(weights, x):
    """Noise-free reverse_linear circuit used for training and noise-free eval."""
    _apply_feature_map(x, p=None)
    _apply_ansatz(weights, p=None)
    return _measurement()


# --- Noisy evaluation circuit factory (frozen weights) ---

def make_noisy_circuit(p):
    """Return a QNode for evaluating FROZEN weights at depolarizing rate p.

    p == 0.0 -> lightning.qubit statevector (exact noise-free baseline), routed
                through the SAME device as the noise-free training circuit so the
                p=0 result is bit-identical.
    p  > 0.0 -> default.mixed density matrix with DepolarizingChannel(p) inserted
                after every gate (feature map + ansatz). interface="numpy" — this
                is the published frozen-weight evaluation path.

    Args:
        p: Per-gate depolarizing error probability in [0, 1].

    Returns:
        Callable QNode: circuit(weights_2d, x) -> scalar expectation value.
    """
    if p == 0.0:
        dev = qml.device(config.QNN_CONFIG["device_train"], wires=NUM_QUBITS)

        @qml.qnode(dev, interface="numpy")
        def _noisefree(weights, x):
            _apply_feature_map(x, p=None)
            _apply_ansatz(weights, p=None)
            return _measurement()

        return _noisefree

    dev = qml.device(config.QNN_CONFIG["device_eval"], wires=NUM_QUBITS)

    @qml.qnode(dev, interface="numpy")
    def _noisy(weights, x):
        _apply_feature_map(x, p=p)
        _apply_ansatz(weights, p=p)
        return _measurement()

    return _noisy


def make_trainable_noisy_circuit(p):
    """Return a TRAINABLE density-matrix QNode at depolarizing rate p (Q5).

    This is the only structural change Q5 introduces. It is gate-for-gate and
    channel-for-channel identical to ``make_noisy_circuit(p)`` (both call the
    shared ``_apply_feature_map`` / ``_apply_ansatz`` primitives with the same
    ``p``), but uses the autograd interface and a differentiable diff_method so
    the noisy circuit can be OPTIMIZED, not just evaluated. The structural gate
    (tests/test_noise_aware.py) asserts this circuit and make_noisy_circuit(p)
    return identical expectation values for identical inputs, so noise-aware
    training and evaluation share exactly one noise model.

    diff_method is config.NOISE_AWARE_DIFF_METHOD ("backprop" by default; fall
    back to "parameter-shift" if backprop misbehaves with L-BFGS-B).

    Args:
        p: Per-gate depolarizing error probability; must be > 0.

    Returns:
        Callable QNode: circuit(weights_2d, x) -> scalar expectation value,
        differentiable w.r.t. weights via qml.grad.
    """
    if p <= 0.0:
        raise ValueError(
            "make_trainable_noisy_circuit requires p > 0; use the noise-free "
            "`circuit` for p=0 training."
        )

    dev = qml.device(config.QNN_CONFIG["device_eval"], wires=NUM_QUBITS)

    @qml.qnode(dev, interface="autograd",
               diff_method=config.NOISE_AWARE_DIFF_METHOD)
    def _trainable_noisy(weights, x):
        _apply_feature_map(x, p=p)
        _apply_ansatz(weights, p=p)
        return _measurement()

    return _trainable_noisy


def make_density_circuit(p):
    """Density-matrix QNode that ALWAYS inserts DepolarizingChannel(p).

    Unlike ``make_noisy_circuit``, this never takes the lightning shortcut at
    p=0 — it builds the default.mixed circuit with explicit channels even when
    p == 0.0. Used by the correctness gate to verify channel-at-zero identity
    (DepolarizingChannel(0) must be the identity, so this must equal the
    noise-free statevector circuit within 1e-10).
    """
    dev = qml.device(config.QNN_CONFIG["device_eval"], wires=NUM_QUBITS)

    @qml.qnode(dev, interface="numpy")
    def _density(weights, x):
        _apply_feature_map(x, p=p)
        _apply_ansatz(weights, p=p)
        return _measurement()

    return _density


def count_depolarizing_channels(p):
    """Count DepolarizingChannel insertions in the reverse_linear noisy circuit.

    Builds the noisy gate sequence (feature map + ansatz) into a QuantumTape at
    the configured reps and tallies every ``DepolarizingChannel``. Used by the
    import-time fidelity assertion below and by src/diagnose_noise.py.

    At reps=3 this must be exactly 42:
        feature map  : 4 qubits * (H + RZ)        = 8
        initial RY   : 4                           = 4
        3 reps * (3 CNOTs * 2 + 4 RY = 10)         = 30
                                              total = 42
    """
    import numpy as _np

    weights = _np.zeros(WEIGHT_SHAPE)
    x = _np.zeros(NUM_QUBITS)
    with qml.tape.QuantumTape() as tape:
        _apply_feature_map(x, p=p)
        _apply_ansatz(weights, p=p)
    return sum(1 for op in tape.operations
               if op.name == "DepolarizingChannel")


# Fail loudly at import if the parameter count is ever structurally wrong.
assert NUM_PARAMS == 16, f"Expected 16 parameters, got {NUM_PARAMS}"
assert config.CNOT_COUNT == len(CNOT_PAIRS) * ANSATZ_REPS == 9, (
    f"Expected 9 CNOTs for reverse_linear, got {config.CNOT_COUNT}"
)

# Fail loudly if the noise injection ever drifts from the published rule: a
# DepolarizingChannel after every H, RZ, RY (one per qubit) and after BOTH the
# control and target of every reverse_linear CNOT -> exactly 42 at reps=3. This
# is the precondition for reproducing the reverse_linear noise-table anchors.
_EXPECTED_DEPOL_COUNT = 42
_depol_count = count_depolarizing_channels(0.01)
assert _depol_count == _EXPECTED_DEPOL_COUNT, (
    f"reverse_linear noisy circuit has {_depol_count} DepolarizingChannel "
    f"insertions at reps={ANSATZ_REPS}, expected {_EXPECTED_DEPOL_COUNT}. "
    f"The noise model diverged from the original rule — the Q5 comparison "
    f"would be INVALID."
)

from qnnbench._selection_circuit import make_circuit
from qnnbench._capacity_circuit import make_train_circuit, weight_shape, num_params, num_cnots
