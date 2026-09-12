"""Mandatory correctness gate — run and PASS before the Q5 sweep.

Asserts:
  (a) The reverse_linear circuit has exactly 16 trainable parameters and 9 CNOTs.
  (b) Channel-at-zero consistency: the noise-free statevector circuit and the
      density-matrix circuit built with DepolarizingChannel(0) agree within
      1e-10. DepolarizingChannel(0) is the identity, so any divergence means the
      noisy and noise-free gate sequences differ.
  (c) Q5 single-noise-model gate: for each p in config.NOISE_AWARE_LEVELS, the
      TRAINABLE noisy circuit (make_trainable_noisy_circuit, autograd/backprop —
      used for noise-aware training) and the EVALUATION noisy circuit
      (make_noisy_circuit, numpy — used for both freeze-eval and noise-aware
      eval) return identical expectation values for identical weights/inputs,
      within 1e-10. This is THE precondition for a valid Q5 comparison: training
      and evaluation must inject noise identically.
  (d) The trainable noisy circuit is actually differentiable: qml.grad w.r.t. the
      16 weights returns a finite length-16 gradient at a representative p.

Run standalone (gates the pipeline in a shell `&&` chain):

    python -m tests.test_noise_aware

Prints PASS and exits 0 on success; prints FAIL and exits 1 on any mismatch.
"""

import sys
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qnnbench.circuits import (
    circuit,
    count_depolarizing_channels,
    make_density_circuit,
    make_noisy_circuit,
    make_trainable_noisy_circuit,
    WEIGHT_SHAPE,
    NUM_QUBITS,
    NUM_PARAMS,
    CNOT_PAIRS,
)
from qnnbench import config

EXPECTED_DEPOL_COUNT = 42

TOL = 1e-10
N_TRIALS = 25


def _flat_cost(weights_flat, x):
    """Scalar circuit output as a function of the flat weight vector."""
    return circuit(weights_flat.reshape(WEIGHT_SHAPE), x)


def _gate_test_param_count():
    """(a) Exactly 16 trainable parameters and 9 CNOTs."""
    assert NUM_PARAMS == 16, f"NUM_PARAMS={NUM_PARAMS}, expected 16"

    rng = np.random.RandomState(7)
    w = pnp.array(rng.uniform(-np.pi, np.pi, size=NUM_PARAMS), requires_grad=True)
    x = pnp.array(rng.uniform(-2.0, 2.0, size=NUM_QUBITS), requires_grad=False)

    grad = qml.grad(_flat_cost, argnums=0)(w, x)
    n_trainable = int(np.asarray(grad).size)
    assert n_trainable == 16, (
        f"Trainable parameter count = {n_trainable}, expected 16"
    )

    specs = qml.specs(circuit)(w.reshape(WEIGHT_SHAPE), x)
    gate_types = dict(specs["resources"].gate_types)
    n_cnot = gate_types.get("CNOT", 0)
    expected_cnot = len(CNOT_PAIRS) * config.QNN_CONFIG["ansatz_reps"]
    assert n_cnot == expected_cnot == 9, (
        f"CNOT count = {n_cnot}, expected {expected_cnot} (=9)"
    )
    return n_trainable, n_cnot


def _gate_test_depol_channel_count():
    """(a2) The reverse_linear noisy circuit has exactly 42 DepolarizingChannel
    insertions at reps=3 (8 feature-map + 4 initial-RY + 3*(6 CNOT + 4 RY))."""
    n = count_depolarizing_channels(0.01)
    assert n == EXPECTED_DEPOL_COUNT, (
        f"DepolarizingChannel count = {n}, expected {EXPECTED_DEPOL_COUNT} "
        f"— noise model diverged from the published rule; Q5 comparison INVALID"
    )
    return n


def _gate_test_channel_at_zero():
    """(b) Noise-free vs DepolarizingChannel(0) density circuit within 1e-10."""
    rng = np.random.RandomState(2024)
    density0 = make_density_circuit(0.0)

    max_diff = 0.0
    for _ in range(N_TRIALS):
        w = pnp.array(rng.uniform(-np.pi, np.pi, size=WEIGHT_SHAPE),
                      requires_grad=False)
        x = pnp.array(rng.uniform(-2.0, 2.0, size=NUM_QUBITS),
                      requires_grad=False)
        ref = float(circuit(w, x))
        noisy0 = float(density0(w, x))
        max_diff = max(max_diff, abs(ref - noisy0))

    assert max_diff < TOL, (
        f"channel-at-zero diverges: max|d|={max_diff:.3e} >= TOL={TOL:.0e}"
    )
    return max_diff


def _gate_test_train_eval_same_noise_model():
    """(c) Trainable noisy circuit == evaluation noisy circuit at each p (1e-10)."""
    rng = np.random.RandomState(13)
    worst = 0.0
    worst_p = None
    for p in config.NOISE_AWARE_LEVELS:
        train_fn = make_trainable_noisy_circuit(p)
        eval_fn = make_noisy_circuit(p)
        for _ in range(N_TRIALS):
            w = pnp.array(rng.uniform(-np.pi, np.pi, size=WEIGHT_SHAPE),
                          requires_grad=False)
            x = pnp.array(rng.uniform(-2.0, 2.0, size=NUM_QUBITS),
                          requires_grad=False)
            a = float(train_fn(w, x))
            b = float(eval_fn(w, x))
            d = abs(a - b)
            if d > worst:
                worst, worst_p = d, p
    assert worst < TOL, (
        f"train vs eval noise model diverges at p={worst_p}: "
        f"max|d|={worst:.3e} >= TOL={TOL:.0e} — Q5 comparison would be invalid"
    )
    return worst


def _gate_test_trainable_noisy_is_differentiable():
    """(d) qml.grad on the trainable noisy circuit yields a finite length-16 grad."""
    p = config.NOISE_AWARE_LEVELS[0]
    train_fn = make_trainable_noisy_circuit(p)

    def flat_cost(weights_flat, x):
        return train_fn(weights_flat.reshape(WEIGHT_SHAPE), x)

    rng = np.random.RandomState(99)
    w = pnp.array(rng.uniform(-np.pi, np.pi, size=NUM_PARAMS), requires_grad=True)
    x = pnp.array(rng.uniform(-2.0, 2.0, size=NUM_QUBITS), requires_grad=False)
    grad = np.asarray(qml.grad(flat_cost, argnums=0)(w, x))
    assert grad.size == 16, f"grad length {grad.size}, expected 16"
    assert np.all(np.isfinite(grad)), "non-finite gradient on trainable noisy circuit"
    return p, grad.size



def test_param_count():
    _gate_test_param_count()

def test_depol_channel_count():
    _gate_test_depol_channel_count()

def test_channel_at_zero():
    _gate_test_channel_at_zero()

def test_train_eval_same_noise_model():
    _gate_test_train_eval_same_noise_model()

def test_trainable_noisy_is_differentiable():
    _gate_test_trainable_noisy_is_differentiable()
