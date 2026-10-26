# python Permutation_Operation.py
# Requirements: Qiskit, Matplotlib
# This script builds a quantum circuit to illustrate the QEF permutation layer.
# It uses SWAP gates to rearrange (permute) the positions of qubits (pixels).

from qiskit import QuantumCircuit

# Define 3 qubits (e.g., representing 3 pixel positions)
qc = QuantumCircuit(3)

# (Optional) Initialize or label qubits as pixel states if needed.
# For demonstration, we assume each qubit holds a pixel's value; initially all |0>.

# Apply swap gates to permute the qubits
qc.swap(0, 1)  # Swap the states of qubit 0 and qubit 1
# After this, the values of pixel0 and pixel1 are exchanged.
qc.swap(1, 2)  # Swap the states of qubit 1 and qubit 2 (formerly pixel0's value moves to qubit2, etc.)
# The two swaps together implement a permutation: original qubit0's state moves to qubit2, qubit1's to qubit0, qubit2's to qubit1.

# Draw and save the quantum circuit diagram
qc.draw(output='mpl', filename='permutation_circuit.png')
print("Permutation circuit drawn and saved as permutation_circuit.png")
