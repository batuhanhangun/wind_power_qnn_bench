"""Fig 1: 4-qubit reverse_linear QNN circuit diagram (D=3), drawn from the
same gate sequence as data/qnn_circuit_source.py."""
import matplotlib.pyplot as plt
import pennylane as qml

from figures.style import apply_style, save

NUM_QUBITS = 4
ANSATZ_REPS = 3
CNOT_PAIRS = [(2, 3), (1, 2), (0, 1)]  # reverse_linear
WIRES = [f"$q_{i}$" for i in range(NUM_QUBITS)]

dev = qml.device("default.qubit", wires=WIRES)


@qml.qnode(dev)
def circuit(weights, x):
    # ZFeatureMap: H then RZ(2x) per qubit.
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=WIRES[i])
        qml.RZ(2.0 * x[i], wires=WIRES[i])
    # Initial RY layer.
    for i in range(NUM_QUBITS):
        qml.RY(weights[0, i], wires=WIRES[i])
    # reps x [reverse_linear CNOT block + RY layer].
    for rep in range(1, ANSATZ_REPS + 1):
        for (c, t) in CNOT_PAIRS:
            qml.CNOT(wires=[WIRES[c], WIRES[t]])
        for i in range(NUM_QUBITS):
            qml.RY(weights[rep, i], wires=WIRES[i])
    return qml.expval(
        qml.PauliZ(WIRES[0]) @ qml.PauliZ(WIRES[1])
        @ qml.PauliZ(WIRES[2]) @ qml.PauliZ(WIRES[3])
    )


def main():
    apply_style()
    import numpy as np
    weights = np.zeros((ANSATZ_REPS + 1, NUM_QUBITS))
    x = np.zeros(NUM_QUBITS)
    # decimals=None -> gate labels stay compact ("RY", "RZ"), no numbers.
    fig, ax = qml.draw_mpl(circuit, style="black_white", decimals=None,
                           show_all_wires=True)(weights, x)
    fig.set_size_inches(6.0, 2.2)
    save(fig, "fig1_qnn_arch")
    plt.close(fig)


if __name__ == "__main__":
    main()
