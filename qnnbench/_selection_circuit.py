"""Parameterized QNN circuit for the structure-selection study.

A single circuit factory produces all three candidate architectures by changing
only the CNOT entanglement pattern of the RealAmplitudes ansatz. All three share
the Z feature map and have exactly 16 parameters (CNOTs are unparameterized), so
the comparison isolates entanglement topology with no capacity confound.

The 'circular' entanglement is gate-for-gate identical to the reference circuit
in ``src/run_qnn.py`` (the manuscript incumbent, QNN-3); this is enforced by
``tests/test_circuit_equivalence.py``.
"""

import sys
from pathlib import Path

import pennylane as qml

# Allow `import config` from the project root.
from qnnbench import config
from qnnbench.config import CONFIGS, ARCHITECTURE_PAIRS as ENTANGLEMENT_PAIRS

NUM_QUBITS = config.QNN_CONFIG["num_qubits"]      # 4
ANSATZ_REPS = config.QNN_CONFIG["ansatz_reps"]    # 3
WEIGHT_SHAPE = (ANSATZ_REPS + 1, NUM_QUBITS)      # (4, 4)
NUM_PARAMS = WEIGHT_SHAPE[0] * WEIGHT_SHAPE[1]    # 16

_dev = qml.device(config.QNN_CONFIG["device"], wires=NUM_QUBITS)


def make_circuit(entanglement):
    """Build a QNode for the given entanglement topology.

    Gate sequence (matching ``src/run_qnn.py`` for ``circular``):
      1. Feature map ZFeatureMap(4, reps=1): Hadamard then RZ(2*x) on each qubit.
      2. Initial RY layer (no entangling before the first layer).
      3. ANSATZ_REPS repetitions of [CNOT pattern, then RY layer].
      4. Measurement of PauliZ on all four qubits as a tensor product.

    Args:
        entanglement: one of ``circular``, ``full``, ``reverse_linear``.

    Returns:
        A PennyLane QNode ``circuit(weights, x)`` where ``weights`` has shape
        WEIGHT_SHAPE and ``x`` has length NUM_QUBITS.
    """
    if entanglement not in ENTANGLEMENT_PAIRS:
        raise ValueError(
            f"Unknown entanglement '{entanglement}'; "
            f"expected one of {CONFIGS}"
        )
    cnot_pairs = ENTANGLEMENT_PAIRS[entanglement]

    @qml.qnode(_dev, interface="autograd",
               diff_method=config.QNN_CONFIG["diff_method"])
    def circuit(weights, x):
        # === Feature map: ZFeatureMap(feature_dimension=4, reps=1) ===
        for i in range(NUM_QUBITS):
            qml.Hadamard(wires=i)
            qml.RZ(2.0 * x[i], wires=i)

        # === Ansatz: RealAmplitudes(4, reps=3, entanglement=<pattern>) ===
        # Initial RY layer (no entanglement before the first layer).
        for i in range(NUM_QUBITS):
            qml.RY(weights[0, i], wires=i)

        # reps repetitions of [CNOT pattern + RY layer].
        for rep in range(1, ANSATZ_REPS + 1):
            for (ctrl, tgt) in cnot_pairs:
                qml.CNOT(wires=[ctrl, tgt])
            for i in range(NUM_QUBITS):
                qml.RY(weights[rep, i], wires=i)

        # === Measurement: Z (x) Z (x) Z (x) Z ===
        return qml.expval(
            qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
        )

    return circuit


# All three candidates have exactly 16 parameters; assert at import time so a
# structural mistake fails loudly rather than silently changing capacity.
assert NUM_PARAMS == 16, f"Expected 16 parameters, got {NUM_PARAMS}"
for _ent in CONFIGS:
    # Parameter count depends only on the RY layers, never on the CNOT pattern.
    assert (ANSATZ_REPS + 1) * NUM_QUBITS == 16, (
        f"Config '{_ent}' would not have 16 parameters"
    )
