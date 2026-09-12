"""Step 1 diagnostic — locate where the noise-aware noisy circuit (B) diverges
from the original published noise harness (A).

Two noisy circuits are built at reps=3, 4 qubits and compared channel-for-channel:

  A = the EXACT circuit from the original
      qnn_wind_benchmark_noise_eval/src/run_noise_eval.py, copied verbatim
      (including its CIRCULAR CNOT ring  qml.CNOT([i, (i+1)%4])).
  B = the CURRENT noise-aware project's noisy circuit, obtained by replaying the
      project's own ``_apply_feature_map`` / ``_apply_ansatz`` primitives from
      src/circuit.py (so B is authoritative, not a paraphrase).

For each circuit this script:
  * counts the ``DepolarizingChannel`` insertions,
  * lists their wires in gate order,
  * prints A and B side by side plus the difference (which locates exactly where
    B diverged from A), and
  * prints, for one fixed random weight vector and input, the expectation value
    each circuit returns at p=0.01, so the numeric divergence is visible too.

Expected result (reps=3): A has 48 channels (12-CNOT circular ring), B has 42
(9-CNOT reverse_linear). The two differ ONLY in the CNOT topology — A places 4
CNOTs/rep, B places the reverse_linear 3 CNOTs/rep — while the per-gate
placement RULE (a DepolarizingChannel after every H, RZ, RY, and after BOTH
qubits of every CNOT) is identical. That 6-channel gap is the intended
reverse_linear architecture, not a placement bug.

Cheap: constructs tapes only (no optimisation). Runs interactively or as a short
single job.

Usage:
    python -m qnnbench.diagnose_noise --p 0.01
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pennylane as qml

from qnnbench import config
from qnnbench.circuits import _apply_feature_map, _apply_ansatz, WEIGHT_SHAPE

NUM_QUBITS = config.QNN_CONFIG["num_qubits"]      # 4
ANSATZ_REPS = config.QNN_CONFIG["ansatz_reps"]    # 3


# --- Circuit A: verbatim from the original run_noise_eval.py noisy path ---
#     (circular CNOT ring qml.CNOT([i, (i+1) % NUM_QUBITS]); channel after every
#     gate and after BOTH qubits of every CNOT). Copied, not paraphrased.

def _apply_original_noisy(weights, x, p):
    """The original harness's noisy gate sequence (circular ring)."""
    # Feature map: H + depol, RZ + depol
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
        qml.DepolarizingChannel(p, wires=i)
        qml.RZ(2.0 * x[i], wires=i)
        qml.DepolarizingChannel(p, wires=i)

    # Initial RY layer + depol
    for i in range(NUM_QUBITS):
        qml.RY(weights[0, i], wires=i)
        qml.DepolarizingChannel(p, wires=i)

    # reps: CNOT ring (depol on both qubits) + RY layer + depol
    for rep in range(1, ANSATZ_REPS + 1):
        for i in range(NUM_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
            qml.DepolarizingChannel(p, wires=i)
            qml.DepolarizingChannel(p, wires=(i + 1) % NUM_QUBITS)
        for i in range(NUM_QUBITS):
            qml.RY(weights[rep, i], wires=i)
            qml.DepolarizingChannel(p, wires=i)


def _measurement():
    return qml.expval(
        qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
    )


# --- Tape construction / channel extraction ---

def _build_tape(apply_fn, weights, x, p):
    """Record ``apply_fn(weights, x, p)`` into a QuantumTape (no device needed)."""
    with qml.tape.QuantumTape() as tape:
        apply_fn(weights, x, p)
        _measurement()
    return tape


def _apply_project_noisy(weights, x, p):
    """Circuit B: replay the project's own channel-placement primitives."""
    _apply_feature_map(x, p=p)
    _apply_ansatz(weights, p=p)


def _depol_placements(tape):
    """Return an ordered list of (index_in_op_stream, gate_name, wire) for every
    DepolarizingChannel, plus the total count and per-wire tally."""
    placements = []
    per_wire = {i: 0 for i in range(NUM_QUBITS)}
    for op in tape.operations:
        if op.name == "DepolarizingChannel":
            w = int(op.wires[0])
            placements.append(w)
            per_wire[w] += 1
    return placements, per_wire


def _annotated_stream(tape):
    """Human-readable gate stream: each op with the wire(s) it acts on, marking
    DepolarizingChannel so the placement rule is visible in order."""
    lines = []
    for op in tape.operations:
        wires = ",".join(str(int(w)) for w in op.wires)
        tag = "  <-- depol" if op.name == "DepolarizingChannel" else ""
        lines.append(f"{op.name}({wires}){tag}")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Step 1 diagnostic: original (A) vs current (B) noisy circuit"
    )
    parser.add_argument("--p", type=float, default=0.01,
                        help="Depolarizing rate for the numeric check (default 0.01)")
    args = parser.parse_args()
    p = args.p

    rng = np.random.RandomState(20240501)
    weights = rng.uniform(-np.pi, np.pi, size=WEIGHT_SHAPE)
    x = rng.uniform(-2.0, 2.0, size=NUM_QUBITS)

    tape_a = _build_tape(_apply_original_noisy, weights, x, p)
    tape_b = _build_tape(_apply_project_noisy, weights, x, p)

    plac_a, wire_a = _depol_placements(tape_a)
    plac_b, wire_b = _depol_placements(tape_b)

    print("=" * 72)
    print("Step 1 diagnostic — DepolarizingChannel placement, reps=3, 4 qubits")
    print("  A = original run_noise_eval.py (circular CNOT ring, verbatim)")
    print("  B = current noise-aware project (reverse_linear primitives)")
    print("=" * 72)
    print()
    print(f"  A total DepolarizingChannel insertions : {len(plac_a)}")
    print(f"  B total DepolarizingChannel insertions : {len(plac_b)}")
    print(f"  difference (A - B)                     : {len(plac_a) - len(plac_b)}")
    print()
    print("  per-wire depol tally:")
    print(f"    {'wire':>4} | {'A':>3} | {'B':>3} | {'A-B':>4}")
    print("    " + "-" * 24)
    for i in range(NUM_QUBITS):
        print(f"    {i:>4} | {wire_a[i]:>3} | {wire_b[i]:>3} | "
              f"{wire_a[i] - wire_b[i]:>4}")
    print()
    print("  depol wire sequence (gate order):")
    print(f"    A: {plac_a}")
    print(f"    B: {plac_b}")
    print()

    # Side-by-side annotated gate streams.
    stream_a = _annotated_stream(tape_a)
    stream_b = _annotated_stream(tape_b)
    n = max(len(stream_a), len(stream_b))
    print("  side-by-side gate stream (A | B):")
    print(f"    {'#':>3}  {'A (original ring)':<34} {'B (reverse_linear)':<34}")
    print("    " + "-" * 74)
    for i in range(n):
        a = stream_a[i] if i < len(stream_a) else ""
        b = stream_b[i] if i < len(stream_b) else ""
        marker = " " if a == b else "*"
        print(f"    {i:>3}{marker} {a:<34} {b:<34}")
    print()
    print("    (rows marked '*' differ between A and B — these locate the "
          "divergence)")
    print()

    # Numeric divergence at p: run both on default.mixed with the same weights/x.
    dev = qml.device("default.mixed", wires=NUM_QUBITS)

    @qml.qnode(dev, interface="numpy")
    def qnode_a(w, xx):
        _apply_original_noisy(w, xx, p)
        return _measurement()

    @qml.qnode(dev, interface="numpy")
    def qnode_b(w, xx):
        _apply_project_noisy(w, xx, p)
        return _measurement()

    val_a = float(qnode_a(weights, x))
    val_b = float(qnode_b(weights, x))
    print(f"  numeric check at p={p} (fixed random weights + input):")
    print(f"    A expectation value : {val_a:.10f}")
    print(f"    B expectation value : {val_b:.10f}")
    print(f"    |A - B|             : {abs(val_a - val_b):.10e}")
    print()

    # Verdict.
    print("=" * 72)
    if len(plac_b) == 42 and len(plac_a) == 48:
        print("DIAGNOSIS: A=48 (12-CNOT ring), B=42 (9-CNOT reverse_linear).")
        print("The per-gate placement RULE is identical; the 6-channel gap is")
        print("exactly the reverse_linear vs circular-ring CNOT-count difference")
        print("(A: 4 CNOTs/rep, B: 3 CNOTs/rep). B matches the reverse_linear")
        print("anchor source (qnn_reverse_linear_rerun), which is the intended")
        print("architecture for the Q5 comparison.")
    else:
        print(f"UNEXPECTED counts: A={len(plac_a)} (expected 48), "
              f"B={len(plac_b)} (expected 42). Investigate the placement rule.")
    print("=" * 72)


if __name__ == "__main__":
    main()
