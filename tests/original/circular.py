import pennylane as qml
from qnnbench import config
NUM_QUBITS = config.QNN_CONFIG["num_qubits"]
ANSATZ_REPS = config.QNN_CONFIG["ansatz_reps"]
WEIGHT_SHAPE = (ANSATZ_REPS + 1, NUM_QUBITS)  # (4, 4) = 16 params
dev = qml.device(config.QNN_CONFIG["device"], wires=NUM_QUBITS)
@qml.qnode(dev, interface="autograd", diff_method=config.QNN_CONFIG["diff_method"])
def circuit(weights, x):
    """Quantum circuit matching Qiskit's ZFeatureMap + RealAmplitudes + Z^4.

    Args:
        weights: Trainable parameters, shape (ansatz_reps+1, num_qubits)
        x: Input features, length num_qubits

    Returns:
        Expectation value of Z x Z x Z x Z (scalar in [-1, 1])
    """
    # === Feature Map: ZFeatureMap(feature_dimension=4, reps=1) ===
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(2.0 * x[i], wires=i)

    # === Ansatz: RealAmplitudes(num_qubits=4, reps=3, entanglement='circular') ===
    # Initial RY layer (no entanglement before first layer)
    for i in range(NUM_QUBITS):
        qml.RY(weights[0, i], wires=i)

    # reps repetitions of [CNOT circular ring + RY layer]
    for rep in range(1, ANSATZ_REPS + 1):
        # Circular CNOT entanglement: (0,1), (1,2), (2,3), (3,0)
        for i in range(NUM_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])
        # RY rotation layer
        for i in range(NUM_QUBITS):
            qml.RY(weights[rep, i], wires=i)

    # === Measurement: Z^{x4} ===
    return qml.expval(
        qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
    )
